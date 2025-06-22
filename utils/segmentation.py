import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def run_full_analysis(file_path, n_clusters_chosen, harga_produk, tujuan_kampanye):
    """
    Menjalankan seluruh alur analisis segmentasi pelanggan, termasuk semua modul AI.
    
    Args:
        file_path (str): Path ke file CSV data.
        n_clusters_chosen (int): Jumlah klaster yang dipilih.
        harga_produk (float): Harga produk dari input pengguna.
        tujuan_kampanye (str): Tujuan kampanye dari input pengguna.
        
    Returns:
        dict: Sebuah dictionary komprehensif berisi semua hasil analisis.
    """
    results = {}
    
    # --- 1. Persiapan Data ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return {"error": f"File tidak ditemukan di path: {file_path}"}

    # --- 2. Pra-pemrosesan ---
    fitur_numerik = ['Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer', 'Num_of_Purchases', 'Average_Transaction_Amount', 'Num_of_Returns', 'Num_of_Support_Contacts', 'Satisfaction_Score', 'Last_Purchase_Days_Ago']
    fitur_kategorikal = ['Gender', 'Email_Opt_In', 'Promotion_Response', 'Target_Churn']
    
    # Penanganan jika kolom tidak ada
    for col in fitur_numerik + fitur_kategorikal:
        if col not in df.columns:
            return {"error": f"Kolom yang dibutuhkan '{col}' tidak ditemukan di dalam file CSV Anda."}

    scaler = StandardScaler()
    df_scaled_numerik = scaler.fit_transform(df[fitur_numerik])
    df_scaled_numerik = pd.DataFrame(df_scaled_numerik, columns=fitur_numerik, index=df.index)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_encoded_kategorikal = encoder.fit_transform(df[fitur_kategorikal])
    df_encoded_kategorikal = pd.DataFrame(df_encoded_kategorikal, columns=encoder.get_feature_names_out(fitur_kategorikal), index=df.index)

    df_processed = pd.concat([df_scaled_numerik, df_encoded_kategorikal], axis=1)

    # --- 3. Evaluasi & Klasterisasi ---
    inertia = []
    silhouette_scores = []
    range_n_clusters = range(2, 11)
    for n in range_n_clusters:
        kmeans_eval = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans_eval.fit(df_processed)
        inertia.append(kmeans_eval.inertia_)
        silhouette_scores.append(silhouette_score(df_processed, kmeans_eval.labels_))
            
    fig_eval, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(range_n_clusters, inertia, marker='o')
    ax1.set_title('Metode Elbow'); ax1.set_xlabel('Jumlah Klaster (K)'); ax1.set_ylabel('Inertia'); ax1.grid(True)
    ax2.plot(range_n_clusters, silhouette_scores, marker='o', color='red')
    ax2.set_title('Silhouette Score'); ax2.set_xlabel('Jumlah Klaster (K)'); ax2.set_ylabel('Silhouette Score'); ax2.grid(True)
    results['evaluation_plot'] = fig_eval

    kmeans = KMeans(n_clusters=n_clusters_chosen, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_processed)
    results['df_with_clusters'] = df

    # --- 4. Profiling & Deskripsi ---
    cluster_profiles = df.groupby('Cluster').agg({
        **{k: 'mean' for k in fitur_numerik}, 
        **{k: (lambda x: x.mode()[0] if not x.mode().empty else 'N/A') for k in fitur_kategorikal}
    }).round(2)
    results['cluster_profiles'] = cluster_profiles

    cluster_descriptions = {}
    for i in range(n_clusters_chosen):
        p = cluster_profiles.loc[i]
        desc = (
        f"**Klaster {i}**:\n"
        f"- Usia rata-rata: **{int(p['Age'])} tahun**\n"
        f"- Pendapatan tahunan: **Rp {p['Annual_Income'] / 1_000_000:.2f} Juta**\n"
        f"- Total pengeluaran: **Rp {p['Total_Spend'] / 1_000_000:.2f} Juta**\n"
        f"- Skor kepuasan: **{p['Satisfaction_Score']}**\n"
        f"- Dominan: **{'Pria' if p['Gender'] == 'Pria' else 'Wanita'}**\n"
        f"- Frekuensi belanja: **{'sering' if p['Num_of_Purchases'] > df['Num_of_Purchases'].mean() else 'jarang'}**\n"
        f"- Respons promosi: **'{p['Promotion_Response']}'**\n"
        f"- Status churn: **{'Cenderung Churn' if p['Target_Churn'] else 'Tidak Cenderung Churn'}**")
        cluster_descriptions[i] = desc
    results['cluster_descriptions'] = cluster_descriptions

    # --- 5. Modul AI Lanjutan ---
    
    gnb = GaussianNB().fit(df_processed, df['Cluster'])

    cf_mapping = {"sangat_tinggi": 0.9, "tinggi": 0.7, "sedang": 0.4, "rendah": 0.2, "pasti": 1.0, "hampir_pasti": 0.8, "kemungkinan_besar": 0.6, "mungkin": 0.4, "tidak_tahu": 0.0, "mungkin_tidak": -0.2, "kemungkinan_besar_tidak": -0.6, "hampir_pasti_tidak": -0.8, "pasti_tidak": -1.0}
    
    def calculate_cf_combination(cf1, cf2):
        if cf1 >= 0 and cf2 >= 0: return cf1 + cf2 * (1 - cf1)
        elif cf1 < 0 and cf2 < 0: return cf1 + cf2 * (1 + cf1)
        else: return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))
        
    def hitung_skor_prioritas_cf(profile, harga_produk, tujuan_kampanye):
        skor, cf_daya_beli, cf_spend, cf_kriteria_kampanye = 0, 0.0, 0.0, 0.0
        if profile['Annual_Income'] > harga_produk * 10: skor += 3; cf_daya_beli = cf_mapping["sangat_tinggi"]
        elif profile['Annual_Income'] > harga_produk * 5: skor += 2; cf_daya_beli = cf_mapping["tinggi"]
        else: skor += 1; cf_daya_beli = cf_mapping["sedang"]
        if profile['Total_Spend'] > harga_produk * 2: skor += 2; cf_spend = cf_mapping["tinggi"]
        elif profile['Total_Spend'] > harga_produk: skor += 1; cf_spend = cf_mapping["sedang"]

        if tujuan_kampanye == "meningkatkan penjualan umum":
            if profile['Promotion_Response'] == 'Merespons': skor += 2; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["hampir_pasti"])
            if profile['Satisfaction_Score'] >= 4: skor += 1; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["tinggi"])
        elif tujuan_kampanye == "mencegah pelanggan churn":
            if profile['Target_Churn']: skor += 3; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["pasti"])
            if profile['Last_Purchase_Days_Ago'] > df['Last_Purchase_Days_Ago'].mean(): skor += 2; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["hampir_pasti"])
        elif tujuan_kampanye == "meningkatkan loyalitas pelanggan":
            if not profile['Target_Churn']: skor += 3; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["pasti"])
            if profile['Years_as_Customer'] > df['Years_as_Customer'].mean(): skor += 2; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["hampir_pasti"])
            if profile['Satisfaction_Score'] >= 4: skor += 1; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["tinggi"])
        elif tujuan_kampanye == "mendapatkan pelanggan baru":
            if profile['Promotion_Response'] == 'Merespons': skor += 2; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["hampir_pasti"])
            if profile['Years_as_Customer'] < df['Years_as_Customer'].mean(): skor += 1; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["tinggi"])
        elif tujuan_kampanye == "menjual produk tambahan":
            if profile['Num_of_Purchases'] > df['Num_of_Purchases'].mean(): skor += 2; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["hampir_pasti"])
            if profile['Promotion_Response'] == 'Merespons': skor += 1; cf_kriteria_kampanye = calculate_cf_combination(cf_kriteria_kampanye, cf_mapping["tinggi"])

        cf_total = calculate_cf_combination(calculate_cf_combination(cf_daya_beli, cf_spend), cf_kriteria_kampanye)
        return skor, cf_total

    prioritas_klaster = []
    for cid in range(n_clusters_chosen):
        profile = cluster_profiles.loc[cid]
        skor, cf = hitung_skor_prioritas_cf(profile, harga_produk, tujuan_kampanye)
        prioritas_klaster.append({'Klaster_ID': cid, 'Skor_Prioritas': skor, 'CF_Prioritas': cf, 'Deskripsi': cluster_descriptions[cid]})
    prioritas_klaster.sort(key=lambda x: x['Skor_Prioritas'], reverse=True)
    results['prioritas_klaster'] = prioritas_klaster

    marketing_rules_fc = [
        {"IF": lambda p: p['Annual_Income'] > 10_000_000 and p['Satisfaction_Score'] >= 4, "THEN": "Kirim penawaran eksklusif/produk premium", "DESC": "Cocok untuk pelanggan berpenghasilan tinggi dan puas."},
        {"IF": lambda p: p['Target_Churn'] and p['Last_Purchase_Days_Ago'] > df['Last_Purchase_Days_Ago'].mean(), "THEN": "Luncurkan kampanye retensi dengan diskon menarik", "DESC": "Prioritas tinggi untuk mencegah pelanggan beralih."},
        {"IF": lambda p: p['Promotion_Response'] == 'Merespons' and p['Num_of_Purchases'] < df['Num_of_Purchases'].mean(), "THEN": "Tawarkan promosi untuk mendorong pembelian berulang", "DESC": "Manfaatkan respons promosi untuk meningkatkan frekuensi pembelian."},
        {"IF": lambda p: p['Average_Transaction_Amount'] < df['Average_Transaction_Amount'].mean() and p['Num_of_Purchases'] > df['Num_of_Purchases'].mean(), "THEN": "Fokus pada up-selling atau cross-selling", "DESC": "Dorong peningkatan nilai keranjang belanja."},
    ]
    marketing_rules_bc = {
        "meningkatkan loyalitas pelanggan": {"THEN_IF": lambda p: not p['Target_Churn'] and p['Years_as_Customer'] > df['Years_as_Customer'].mean() and p['Satisfaction_Score'] >= 4, "STRATEGY": "Kirim hadiah loyalitas atau undangan acara eksklusif."},
        "mencegah pelanggan churn": {"THEN_IF": lambda p: p['Target_Churn'] and p['Last_Purchase_Days_Ago'] > df['Last_Purchase_Days_Ago'].mean(), "STRATEGY": "Luncurkan kampanye retensi agresif."},
        "mendapatkan pelanggan baru": {"THEN_IF": lambda p: p['Promotion_Response'] == 'Merespons' and p['Years_as_Customer'] < df['Years_as_Customer'].mean(), "STRATEGY": "Iklankan melalui kanal yang menarik demografi klaster ini."},
    }
    
    fc_results = {}
    for item in prioritas_klaster:
        cid = item['Klaster_ID']; profile = cluster_profiles.loc[cid]; fc_results[cid] = []
        for rule in marketing_rules_fc:
            if rule["IF"](profile): fc_results[cid].append(rule)
    results['forward_chaining_results'] = fc_results
    
    bc_results = []
    if tujuan_kampanye in marketing_rules_bc:
        rule = marketing_rules_bc[tujuan_kampanye]
        for cid in range(n_clusters_chosen):
            profile = cluster_profiles.loc[cid]
            if rule["THEN_IF"](profile):
                bc_results.append({'Klaster_ID': cid, 'Deskripsi': cluster_descriptions[cid], 'Strategi': rule["STRATEGY"]})
    results['backward_chaining_results'] = bc_results
    
    def get_hierarchical_plan(strategy_type):
        plans = {
            "Kirim penawaran eksklusif/produk premium": {"Mayor": "Luncheon Kampanye Produk Premium", "Steps": ["Desain Materi Pemasaran", "Segmentasikan Audiens", "Jadwalkan Pengiriman"]},
            "Luncurkan kampanye retensi dengan diskon menarik": {"Mayor": "Implementasi Program Retensi", "Steps": ["Identifikasi Pelanggan Berisiko", "Rancang Penawaran Personalisasi", "Sertakan Survei Kepuasan"]},
            "Tawarkan promosi untuk mendorong pembelian berulang": {"Mayor": "Optimalisasi Promosi Pembelian Berulang", "Steps": ["Analisis Produk Sering Dibeli", "Buat Kupon/Poin Loyalitas", "Kirim Notifikasi Otomatis"]},
            "Fokus pada up-selling atau cross-selling": {"Mayor": "Strategi Peningkatan Nilai Keranjang", "Steps": ["Identifikasi Produk Komplementer", "Implementasikan Rekomendasi di Situs", "Latih Tim Penjualan"]},
            "Kirim hadiah loyalitas atau undangan acara eksklusif.": {"Mayor": "Program Apresiasi Pelanggan Loyal", "Steps": ["Verifikasi Kriteria Loyalitas", "Pilih Jenis Hadiah/Acara", "Komunikasikan Manfaat Program"]},
            "Luncurkan kampanye retensi agresif.": {"Mayor": "Kampanye Retensi Agresif", "Steps": ["Hubungi Pelanggan Secara Personal", "Tawarkan Diskon 'Win-Back'", "Analisis Feedback Churn"]},
            "Iklankan melalui kanal yang menarik demografi klaster ini.": {"Mayor": "Akuisisi Pelanggan Baru Tertarget", "Steps": ["Analisis Preferensi Media", "Buat Konten Iklan Relevan", "Monitor Kinerja Iklan"]},
        }
        return plans.get(strategy_type, {"Mayor": "Strategi Umum", "Steps": ["Langkah umum 1", "Langkah umum 2"]})
    
    top_strategy = None
    if bc_results: top_strategy = bc_results[0]['Strategi']
    elif fc_results.get(prioritas_klaster[0]['Klaster_ID']): top_strategy = fc_results[prioritas_klaster[0]['Klaster_ID']][0]['THEN']
    
    if top_strategy:
        results['hierarchical_plan'] = get_hierarchical_plan(top_strategy)
        results['top_strategy_for_plan'] = top_strategy

    summary_text = f"""
    Sistem ini berhasil melakukan **Segmentasi Pelanggan E-commerce** menggunakan algoritma K-Means, membagi total **{n_clusters_chosen}** segmen pelanggan unik.
    \n#### 1. Profil Klaster yang Ditemukan:"""
    for cid, desc in cluster_descriptions.items():
        summary_text += f"\n- {desc}"
    summary_text += f"""
    \n\n#### 2. Rekomendasi Target Pasar Utama:
    Berdasarkan harga produk **Rp {harga_produk:,.0f}** dan tujuan kampanye **'{tujuan_kampanye.upper()}'**, klaster berikut direkomendasikan:"""
    for i, item in enumerate(prioritas_klaster):
        summary_text += f"\n- **Prioritas {i+1}: Klaster {item['Klaster_ID']}** (Skor: {item['Skor_Prioritas']}, CF: {item['CF_Prioritas']:.2f})"
    summary_text += """
    \n\n#### 3. Penerapan Konsep Kecerdasan Buatan:
    - **Teorema Bayes**: Model Naive Bayes dilatih untuk memperkirakan probabilitas $P(\text{Klaster} | \text{Fitur})$, memungkinkan penempatan pelanggan baru ke segmen yang paling mungkin.
    - **Certainty Factor (CF)**: Setiap rekomendasi dilengkapi nilai CF yang merefleksikan tingkat keyakinan sistem terhadap relevansi klaster dengan tujuan kampanye.
    - **Inferensi (Forward & Backward Chaining)**:
        - **Forward Chaining (Data-driven):** Sistem secara otomatis menyarankan strategi pemasaran begitu karakteristik klaster diketahui.
        - **Backward Chaining (Goal-driven):** Sistem mencari klaster yang paling mendukung tujuan bisnis yang telah ditentukan.
    - **Perencanaan Hierarkis**: Strategi yang direkomendasikan diuraikan menjadi 'Tindakan Mayor' dan 'Langkah-langkah Minor', memfasilitasi eksekusi yang terstruktur.
    - **Konsep Pencarian (Heuristik & Buta)**: Metode Elbow dan `n_init=10` pada K-Means adalah bentuk pencarian untuk menemukan solusi optimal dan menghindari hasil yang suboptimal.
    - **Konsep Agen & Multi-Agen**: Sistem ini berfungsi sebagai 'Agen Analisis' yang outputnya dapat diintegrasikan ke dalam ekosistem 'Multi-Agen' (misal: Agen Pemasaran Email, Agen Iklan).
    \nSecara keseluruhan, proyek ini mengintegrasikan berbagai konsep inti AI untuk memberikan rekomendasi strategi pemasaran yang lebih cerdas, terjustifikasi, dan terstruktur.
    """
    results['final_summary'] = summary_text
    
    return results
pass

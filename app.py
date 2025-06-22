import streamlit as st
import pandas as pd
from utils.segmentation import run_full_analysis # Impor fungsi analisis terlengkap

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Dashboard AI Segmentasi Pelanggan")

# --- Judul Aplikasi ---
st.title("🚀 Dashboard AI untuk Segmentasi & Strategi Pemasaran")
st.markdown("Unggah data pelanggan Anda untuk mendapatkan analisis mendalam, profil segmen, dan rekomendasi strategi berbasis AI yang dapat ditindaklanjuti.")

uploaded_file = st.file_uploader("Unggah file CSV data pelanggan Anda", type=["csv"])

if uploaded_file is None:
    st.info("👋 Selamat datang! Silakan unggah file CSV data pelanggan melalui panel di sebelah kiri untuk memulai analisis.")
    st.stop()
else:
    file_path = uploaded_file

# --- Sidebar untuk Input ---
st.sidebar.header("⚙️ Panel Kontrol Analisis")

st.sidebar.subheader("Parameter Analisis")
n_clusters = st.sidebar.slider("Pilih Jumlah Klaster (K)", 2, 10, 3, help="Pilih jumlah segmen pelanggan yang ingin Anda buat.")
harga_produk = st.sidebar.number_input("Masukkan Harga Produk (Rp)", 0, None, 150000, 50000)
tujuan_kampanye = st.sidebar.selectbox(
    "Pilih Tujuan Utama Kampanye",
    [
        "meningkatkan loyalitas pelanggan", "mencegah pelanggan churn",
        "mendapatkan pelanggan baru", "meningkatkan penjualan umum", "menjual produk tambahan"
    ],
    index=0, key="tujuan"
).lower()

if st.sidebar.button("🧠 Jalankan Analisis AI Lengkap", type="primary", use_container_width=True):
    with st.spinner("🧙‍♂️ Menganalisis data, menjalankan model AI, menyusun strategi, dan menulis laporan..."):
        hasil = run_full_analysis(file_path, n_clusters, harga_produk, tujuan_kampanye)

    if "error" in hasil:
        st.error(hasil["error"])
    else:
        st.success("✅ Analisis Komprehensif Selesai!")

        # --- Tampilan Hasil dengan Tab ---
        tab_list = ["🎯 Rekomendasi Utama", "👥 Profil Segmen", "🤖 Strategi AI", "📈 Evaluasi Model", "📜 Laporan Lengkap", "📄 Data Hasil"]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)

        with tab1:
            st.header("🎯 Rekomendasi Target Pasar Utama")
            st.markdown(f"Berdasarkan harga produk **Rp {harga_produk:,.0f}** dan tujuan **'{tujuan_kampanye.upper()}'**:")
            for i, item in enumerate(hasil["prioritas_klaster"]):
                st.subheader(f"#{i+1} Prioritas: Klaster {item['Klaster_ID']}")
                col1, col2 = st.columns(2)
                col1.metric("Skor Prioritas", item['Skor_Prioritas'])
                col2.metric("Certainty Factor (CF)", f"{item['CF_Prioritas']:.2f}")
                st.markdown(item['Deskripsi'])
                st.divider()

        with tab2:
            st.header(f"👥 Profil Detail dari {n_clusters} Segmen Pelanggan")
            st.dataframe(hasil["cluster_profiles"], use_container_width=True)
            for cid, desc in hasil["cluster_descriptions"].items():
                with st.expander(f"Lihat deskripsi lengkap untuk **Klaster {cid}**"):
                    st.markdown(desc)

        with tab3:
            st.header("🤖 Rekomendasi Strategi & Perencanaan AI")
            
            st.subheader("Backward Chaining (Goal-Driven)")
            st.markdown(f"Untuk tujuan **'{tujuan_kampanye.upper()}'**, klaster berikut paling relevan:")
            if hasil.get('backward_chaining_results'):
                for item in hasil['backward_chaining_results']:
                    st.success(f"**Klaster {item['Klaster_ID']}**: {item['Strategi']}")
            else:
                st.info("Tidak ada klaster yang cocok secara spesifik dengan tujuan ini melalui Backward Chaining.")
            
            st.subheader("Forward Chaining (Data-Driven)")
            st.markdown("Strategi yang disarankan berdasarkan karakteristik masing-masing klaster:")
            for cid, strategies in hasil['forward_chaining_results'].items():
                if strategies:
                    with st.container(border=True):
                        st.write(f"**Rekomendasi untuk Klaster {cid}**")
                        for s in strategies:
                            st.markdown(f"- **{s['THEN']}**: *{s['DESC']}*")
            
            st.subheader("Hierarchical Planning (Rencana Aksi)")
            if 'hierarchical_plan' in hasil:
                st.markdown(f"Rencana untuk strategi teratas: **'{hasil['top_strategy_for_plan']}'**")
                plan = hasil['hierarchical_plan']
                st.success(f"**Tindakan Mayor:** {plan['Mayor']}")
                st.markdown("**Langkah-langkah Minor:**")
                for i, step in enumerate(plan['Steps']):
                    st.markdown(f"  {i+1}. {step}")
            else:
                st.info("Tidak ada rencana aksi spesifik yang dapat dibuat untuk strategi teratas.")
                
        with tab4:
            st.header("📈 Evaluasi Model Klasterisasi")
            st.pyplot(hasil["evaluation_plot"])
            st.info(f"Klasterisasi dijalankan dengan **K = {n_clusters}**. Grafik di atas (Metode Elbow & Silhouette Score) dapat membantu dalam validasi pemilihan K.")

        with tab5:
            st.header("📜 Laporan Eksekutif & Penjelasan Konsep AI")
            st.markdown(hasil.get('final_summary', "Tidak ada kesimpulan yang dihasilkan."))

        with tab6:
            st.header("📄 Data Hasil dengan Label Klaster")
            df_hasil = hasil["df_with_clusters"]
            st.dataframe(df_hasil, use_container_width=True)
            
            @st.cache_data
            def convert_df(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Unduh Data Hasil (.csv)",
                data=convert_df(df_hasil),
                file_name=f"hasil_segmentasi_{n_clusters}_klaster.csv",
                mime="text/csv",
                use_container_width=True
            )
else:
    st.sidebar.info("Atur parameter di atas dan klik tombol untuk memulai.")


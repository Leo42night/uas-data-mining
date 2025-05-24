import streamlit as st
import pandas as pd
# import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx
import re
import json # load data filter
from datetime import datetime

# --- LOAD DATA ---

# Load JSON (setup Filter)
with open("setup_filter.json", "r") as f:
    loaded_data = json.load(f)

# Konversi kembali min_date & max_date ke datetime.date
loaded_data["min_date"] = datetime.strptime(loaded_data["min_date"], "%Y-%m-%d").date()
loaded_data["max_date"] = datetime.strptime(loaded_data["max_date"], "%Y-%m-%d").date()

# Load dataset yang dibutuhkan
# KPI
spend_chart = pd.read_csv("spend_chart.csv")
bubble_chart = pd.read_csv("bubble_chart.csv")
page_chart = pd.read_csv("page_chart.csv")
line_chart = pd.read_csv("line_chart.csv")
interact_chart = pd.read_csv("interact_chart.csv")
stacked_chart = pd.read_csv("stacked_chart.csv")
city_sales = pd.read_csv("city_sales.csv")
# Model Asosiasi
rules = pd.read_csv("association_rules.csv")  # Pastikan file tersedia
# Model Clustering

st.set_page_config(page_title="Shop Mining Dashboard", layout="wide")

def filter(df):
    # Filter berdasarkan IP dan Tanggal
    filtered_df = df[
        (df['ip_address'].isin(selected_ips)) &
        (pd.to_datetime(df['order_date']).dt.date >= date_range[0]) &
        (pd.to_datetime(df['order_date']).dt.date <= date_range[1])
    ]    
    return filtered_df

# --- SIDEBAR ---
# === Sidebar Logo ===
st.sidebar.image(
    "https://raw.githubusercontent.com/Leo42night/Leo42night/main/img/logo_shopmining.png",
    caption="[datamininguntan.my.id](https://datamininguntan.my.id/)"
)

# === Sidebar Deskripsi ===
st.sidebar.caption("""
Data Mining Online Service adalah layanan untuk memudahkan pengguna dalam melakukan analisis data online. Layanan ini menyediakan fitur-fitur seperti visualisasi data, modelling, serta rekomendasi produk berdasarkan transaksi.
""")

# === Sidebar Navigasi Halaman ===
page = st.sidebar.radio("Pilih Halaman", ["Visualisasi", "Model: Association Rules"])

# ---------------------------------- HALAMAN VISUALISASI ----------------------------------
if page == "Visualisasi":
    # === SIDEBAR FILTER ===
    selected_ips = st.sidebar.multiselect("Filter IP Address", options=loaded_data["ip_options"], default=loaded_data["ip_options"])
    date_range = st.sidebar.date_input("Filter Tanggal Order", value=(loaded_data["min_date"], loaded_data["max_date"]), min_value=loaded_data["min_date"], max_value=loaded_data["max_date"])

    # === VISUALISASI DATA E-COMMERCE ===
    st.header("📊 Visualisasi Data E-commerce")

    # ------------------------------------------------------------------------------------
    # --- KPI Metrics Section ---
    st.subheader("KPI Metrics", divider=True)

    # Buat 4 kolom
    col1, col2, col3, col4 = st.columns(4)
    # Isi masing-masing kolom dengan KPI
    with col1:
        st.metric(label="Transactions", value="110", delta="+5%")

    with col2:
        st.metric(label="Customers", value="96", delta="+3.2%")

    with col3:
        st.metric(label="Products", value="107", delta="+8.1%")

    with col4:
        st.metric(label="Revenue", value="Rp47.068.409", delta="-0.5%")
        
    # ------------------------------------------------------------------------------------
    # Dua kolom untuk Bar & Pie Chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Customer Berdasarkan Total Spend", divider=True)
        
        # filter data
        filtered_df = filter(spend_chart)
        
        # Hitung jumlah customer di setiap kategori
        spend_distribution = filtered_df["spend_category"].value_counts().sort_index()

        fig, ax = plt.subplots()
        sns.barplot(
            x=spend_distribution.index,
            y=spend_distribution.values,
            palette="coolwarm",
            ax=ax,
            hue=spend_distribution.index,
            dodge=False,
        )
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )

        ax.set_xlabel("Range Total Spend (IDR)")
        ax.set_ylabel("Jumlah Customer")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Persentase Customer Berdasarkan Total Spend", divider=True)
        fig2 = plt.figure(figsize=(6, 6))
        plt.pie(
            spend_distribution,
            labels=spend_distribution.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("coolwarm"),
            labeldistance=1.2,
            pctdistance=0.8,
        )
        plt.legend(spend_distribution.index, loc="best", bbox_to_anchor=(1, 1))
        st.pyplot(fig2)
        
    # Dua kolom untuk Bar & Pie Chart
    col3, col4 = st.columns(2)
    
    with col3:
        # ------------------------------------------------------------------------------------
        # Bubble Cloud Chart
        st.subheader("Top 5 Produk by Total Penjualan", divider=True)
        
        # Filter berdasarkan IP dan Tanggal
        filtered_df_bubble = filter(bubble_chart)

        df_top5 = filtered_df_bubble.sort_values('total_sales', ascending=False).head(5)
        sizes = (df_top5['total_sales'] / df_top5['total_sales'].max()) * 100 + 40

        # Atur posisi bubble
        G = nx.Graph()
        for i in range(len(df_top5)):
            G.add_node(i)
        pos = nx.spring_layout(G, k=0.5, seed=42)
        x_pos = [pos[i][0] for i in range(len(df_top5))]
        y_pos = [pos[i][1] for i in range(len(df_top5))]
        
        def wrap_text(text, max_words=4):
            words = text.split()
            return "<br>".join([" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)])

        # Build Plotly figure 
        fig_bubble = go.Figure(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=df_top5['total_sales'],
                colorscale='Viridis',
                showscale=False, # tidak menampilkan skala warna
                line=dict(width=2, color='DarkSlateGrey'),
                sizemode='diameter',
                opacity=0.7,
            ),
            hoverinfo='text',
            hovertext=[f"{name}<br>Total Sales: {total:,.0f}" for name, total in zip(df_top5['order_item_name'], df_top5['total_sales'])]
        ))

        annotations = []
        for x, y, name, total in zip(x_pos, y_pos, df_top5['order_item_name'], df_top5['total_sales']):
            wrapped_text = wrap_text(name)
            full_text = f"{wrapped_text}<br><b>{int(total):,}</b>"
            annotations.append(dict(
                x=x,
                y=y,
                text=full_text,
                showarrow=False,
                font=dict(
                    color="black",
                    size=12,
                    family="Arial"
                ),
                align="center",
                xanchor="center",
                yanchor="middle",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            ))

        fig_bubble.update_layout(annotations=annotations)

        fig_bubble.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600,
            width=800,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig_bubble, use_container_width=True)
    
    with col4:
        # ------------------------------------------------------------------------------------
        st.subheader("Halaman Produk dengan Interaksi Tertinggi", divider=True)
        
        # Filter berdasarkan IP dan Tanggal
        filtered_df_page = filter(page_chart)

        # Hitung jumlah interaksi tiap halaman
        page_counts = filtered_df_page["meta_value"].value_counts().reset_index()
        page_counts.columns = ["page", "interaction_count"]

        # Ambil 10 page teratas
        top_pages = page_counts.head(10).copy()

        # Hitung total interaksi untuk persentase
        total_interactions = top_pages["interaction_count"].sum()
        top_pages["percentage"] = (top_pages["interaction_count"] / total_interactions) * 100

        # Buat donut chart
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            top_pages["percentage"],
            labels=top_pages["page"],
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.85
        )

        # Tambahkan lubang di tengah (donut hole)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        plt.tight_layout() # menyesuaikan letak subplot: tidak bertumpuk atau terlalu rapat
        # Tampilkan di Streamlit
        st.pyplot(fig)
        
    # ------------------------------------------------------------------------------------
    # --- Visualisasi Frekuensi Transaksi per Jam berdasarkan IP ---
    st.subheader("Frekuensi Order by Jam & IP Address", divider=True)

    # Filter berdasarkan IP dan Tanggal
    filtered_df_line = filter(line_chart)

    # Pivot agar IP Address menjadi kolom
    pivot_df = filtered_df_line.pivot(index='order_date', columns='ip_address', values='jumlah_transaksi').fillna(0)

    # Plotting
    fig_line, ax_line = plt.subplots(figsize=(18, 8))
    pivot_df.plot(ax=ax_line, marker='o', markersize=8, linewidth=1.5)
    ax_line.set_title('Frekuensi Pembelian per Jam Berdasarkan IP Address')
    ax_line.set_xlabel('Waktu Pemesanan')
    ax_line.set_ylabel('Jumlah Transaksi')
    ax_line.legend(title='IP Address', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax_line.grid(True)
    plt.tight_layout()
    
    # Plotting
    st.pyplot(fig_line)
    
    # Dua kolom untuk Bar & Pie Chart
    col5, col6 = st.columns(2)
    
    with col5:
        # ------------------------------------------------------------------------------------
        # --- Visualisasi Top 10 Produk dengan Interaksi Terbanyak ---
        st.subheader("Top 10 Produk Berdasarkan Interaksi", divider=True)

        # Ambil 10 produk teratas
        top_products = interact_chart.head(10)

        # Visualisasi bar chart horizontal
        fig, ax = plt.subplots(figsize=(12, 6))
        plot = sns.barplot(
            data=top_products,
            x="interaction_count",
            y="product_name",
            palette="crest",
            hue="product_name",
            legend=False,
            ax=ax
        )

        for bar in plot.containers:
            plot.bar_label(bar, fmt='%d', label_type='edge', padding=3)

        ax.set_title("Produk dengan Interaksi Terbanyak (Top 10)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Jumlah Interaksi")
        ax.set_ylabel("Nama Produk")
        plt.tight_layout()

        st.pyplot(fig)
    with col6:    
        # ------------------------------------------------------------------------------------
        # Stacked Bar Chart
        st.subheader("Customer tiap IP Address & Frekuensi-nya", divider=True)
        
        filtered_stacked_chart = filter(stacked_chart)
        
        # Hitung frekuensi order berdasarkan ip_address dan customer_id
        order_freq = filtered_stacked_chart.groupby(['ip_address', 'customer_id']).size().unstack(fill_value=0)
        
        # Membuat objek figure dan axis
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Plot stacked bar chart
        order_freq.plot(kind='bar', stacked=True, ax=ax, legend=False)

        # Hitung jumlah customer unik per IP
        unique_cust = order_freq.gt(0).sum(axis=1)

        # Tambahkan jumlah customer di atas setiap bar
        totals = order_freq.sum(axis=1)  # untuk tahu tinggi bar
        for i, (cust, total) in enumerate(zip(unique_cust, totals)):
            ax.text(i, total + 0.3, f"{cust} cust", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.ylabel('Frekuensi Order')
        plt.xlabel('IP Address')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Tampilkan plot di Streamlit
        st.pyplot(fig)    
    
    
    # ------------------------------------------------------------------------------------
    # --- Visualisasi Sebaran Transaksi per Kota ---
    st.subheader("Sebaran Transaksi per Kota", divider=True)
    st.markdown("Tiap *Point* dilengkapi Jumlah Transaksi dan Produk Terlaris")

    # Tambahkan spinner agar Streamlit tidak tampak diam saat loading lokasi
    with st.spinner("Menyiapkan visualisasi..."):

        fig_map = px.scatter_geo(
            city_sales,
            lat='latitude',
            lon='longitude',
            size='transaction_count',
            color_discrete_sequence=['red'],
            hover_name='city',
            hover_data={
                'transaction_count': True,
                'most_bought_product': True,
                'latitude': False,
                'longitude': False
            },
            projection='mercator'
        )

        fig_map.update_geos(
            visible=True,
            lataxis_range=[-11, 6],
            lonaxis_range=[95, 141],
            showcountries=True,
            countrycolor="Black"
        )

        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend_title_text='Jumlah Transaksi'
        )

        st.plotly_chart(fig_map, use_container_width=True)


# ---------------------------------- HALAMAN MODELLING 1 ----------------------------------
elif page == "Model: Association Rules":
    st.header("🤖 Modelling - Association Rules")

    st.caption(
        """
        Proyek ini menggunakan **mlxtend.frequent_patterns** untuk mengidentifikasi aturan asosiasi.
        Masukkan produk yang dibeli, sistem akan merekomendasikan produk lain berdasarkan pola pembelian.
        """,
        unsafe_allow_html=True
    )

    product_list = sorted(set(rules["antecedents"].apply(eval).explode()))
    selected_products = st.multiselect("Pilih produk yang dibeli:", product_list)

    new_transaction = set(selected_products)
    matched_rules = rules[rules["antecedents"].apply(lambda x: set(eval(x)).issubset(new_transaction))]

    def clean_frozenset(text):
        cleaned = re.sub(r"frozenset\(\{(.*?)\}\)", r"[\1]", text)
        return cleaned.replace("'", '"')

    matched_rules["antecedents"] = matched_rules["antecedents"].apply(clean_frozenset)
    matched_rules["consequents"] = matched_rules["consequents"].apply(clean_frozenset)

    st.subheader("Rekomendasi Produk Berdasarkan Transaksi")
    st.markdown("""
    - **Jika aturan ditemukan**, produk dalam *consequents* bisa direkomendasikan kepada pengguna.<br>
    - **Confidence** menunjukkan seberapa besar peluang bahwa orang yang membeli antecedent juga akan membeli consequent.<br>
    - **Lift** menunjukkan seberapa kuat hubungan antara antecedent dan consequent dibandingkan pembelian acak.<br>
    """, unsafe_allow_html=True)

    if matched_rules.empty:
        st.write("Tidak ada rekomendasi berdasarkan transaksi ini.")
    else:
        st.dataframe(matched_rules[["antecedents", "consequents", "confidence", "lift"]])

    st.subheader("Visualisasi Confidence dan Lift")
    if not matched_rules.empty:
        fig3, ax3 = plt.subplots()
        sns.barplot(
            x=matched_rules["consequents"],
            y=matched_rules["confidence"],
            color="blue",
            label="Confidence",
            ax=ax3
        )
        sns.barplot(
            x=matched_rules["consequents"],
            y=matched_rules["lift"],
            color="red",
            alpha=0.5,
            label="Lift",
            ax=ax3
        )
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.xlabel("Consequents")
        plt.title("Confidence vs Lift dari Produk Rekomendasi")
        plt.legend()
        st.pyplot(fig3)

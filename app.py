import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="Shop Mining Dashboard", layout="wide")

# --- UTILITIES ---
@st.cache_data(show_spinner="Memuat data transaksi dan koordinat kota...", persist=True)
def load_city_sales(selected_ips, date_range):
    import pandas as pd
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    # Load data utama
    df = pd.read_csv("csv/fact_sales_v2.csv", sep=';', encoding='utf-8')
    df['city'] = df['city'].str.replace(r'^(Kota|Kabupaten)\s+', '', case=False, regex=True).str.strip()
    df = df[df['city'].notnull() & (df['city'] != '')]

    # Gabungkan IP address dari fact_sales_v1
    df_v1 = pd.read_csv("csv/fact_sales_v1.csv", sep=';')
    df_v1 = df_v1[['order_id', 'ip_address']].drop_duplicates()
    df = pd.merge(df, df_v1, on='order_id', how='left')

    # Filter transaksi unik
    unique_orders = df[['order_id', 'city', 'ip_address', 'order_date']].drop_duplicates()
    unique_orders['order_date'] = pd.to_datetime(unique_orders['order_date'])

    # Filter berdasarkan input user
    filtered_orders = unique_orders[
        (unique_orders['ip_address'].isin(selected_ips)) &
        (unique_orders['order_date'].dt.date >= date_range[0]) &
        (unique_orders['order_date'].dt.date <= date_range[1])
    ]

    # Hitung transaksi per kota
    city_transaction_counts = filtered_orders['city'].value_counts().reset_index()
    city_transaction_counts.columns = ['city', 'transaction_count']

    # Produk terlaris per kota
    most_bought_items = (
        df.groupby(['city', 'order_item_name'])
        .size()
        .reset_index(name='count')
        .sort_values(['city', 'count'], ascending=[True, False])
        .drop_duplicates('city')
        .rename(columns={'order_item_name': 'most_bought_product'})
    )

    city_sales = pd.merge(city_transaction_counts, most_bought_items[['city', 'most_bought_product']], on='city', how='left')

    # Geocoding
    geolocator = Nominatim(user_agent="myApp", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    def safe_geocode(city):
        try:
            return geocode(f"{city}, Indonesia")
        except:
            return None

    city_sales['location'] = city_sales['city'].apply(safe_geocode)
    city_sales['latitude'] = city_sales['location'].apply(lambda loc: loc.latitude if loc else None)
    city_sales['longitude'] = city_sales['location'].apply(lambda loc: loc.longitude if loc else None)

    city_sales = city_sales.dropna(subset=['latitude', 'longitude'])
    return city_sales

# --- LOAD DATA ---
# Load data yang dibutuhkan
spend_distribution = pd.read_csv("spend_distribution.csv", index_col=0).iloc[:, 0]
rules = pd.read_csv("association_rules.csv")  # Pastikan file tersedia
sales_v1 = pd.read_csv('csv/fact_sales_v1.csv', sep=';')
sales_v1['order_date'] = pd.to_datetime(sales_v1['order_date'])

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
page = st.sidebar.radio("Pilih Halaman", ["Visualisasi", "Modelling"])

# ---------------------------------- HALAMAN VISUALISASI ----------------------------------
if page == "Visualisasi":
    # === SIDEBAR FILTER ===
    # Load data untuk filter
    df_v1_sidebar = sales_v1.copy(deep=True)
    # IP Address unik
    ip_options = sorted(df_v1_sidebar['ip_address'].dropna().unique())
    selected_ips = st.sidebar.multiselect("Filter IP Address", options=ip_options, default=ip_options)

    # Rentang tanggal
    min_date = df_v1_sidebar['order_date'].min().date()
    max_date = df_v1_sidebar['order_date'].max().date()
    date_range = st.sidebar.date_input("Filter Tanggal Order", value=(min_date, max_date), min_value=min_date, max_value=max_date)


    # === VISUALISASI DATA E-COMMERCE ===
    st.header("📊 Visualisasi Data E-commerce")

    # ------------------------------------------------------------------------------------
    # --- KPI Metrics Section ---
    st.subheader("KPI Metrics")

    # CSS for cards
    card_style = """
    <style>
    .card {
      background-color: #1e1e1e;
      padding: 1rem;
      border-radius: 0.5rem;
      text-align: center;
      margin-bottom: 1rem;
    }
    .card h3 { color: #ffffff; margin: 0; font-size: 1rem; }
    .card h2 { color: #66c0f4; margin: 0; font-size: 1.5rem; }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="card">
          <h3>Total Sales</h3>
          <h2>110</h2>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
          <h3>Total Customers</h3>
          <h2>96</h2>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card">
          <h3>Total Revenue</h3>
          <h2>Rp 47.068.409</h2>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="card">
          <h3>Total Products</h3>
          <h2>107</h2>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------------------------
    # Dua kolom untuk Bar & Pie Chart
    col1, col2 = st.columns(2)

    with col1:
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
        ax.set_title("Distribusi Customer Berdasarkan Total Spend")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        fig2 = plt.figure(figsize=(6, 6))
        plt.pie(
            spend_distribution,
            labels=spend_distribution.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("coolwarm"),
            labeldistance=1.2,
            pctdistance=0.8,
        )
        plt.title("Persentase Customer Berdasarkan Total Spend")
        plt.legend(spend_distribution.index, loc="best", bbox_to_anchor=(1, 1))
        st.pyplot(fig2)

    # Bubble Cloud Chart
    st.subheader("Bubble Cloud Chart: Top 5 Produk by Total Penjualan")

    # Load & preprocess data
    df_sales = pd.read_csv('csv/fact_sales_v2.csv', sep=';')
    if df_sales['total_sales'].dtype == object:
        df_sales['total_sales'] = (
            df_sales['total_sales']
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
    dag_sales = df_sales.groupby('product_id', as_index=False)['total_sales'].sum()

    df_order_product = pd.read_csv('csv/order_product.csv', sep=';')
    df_merge1 = dag_sales.merge(
        df_order_product[['order_item_id', 'product_id']],
        on='product_id', how='left'
    )

    df_items = pd.read_csv('csv/order_items.csv', sep=';')
    df_merge2 = df_merge1.merge(
        df_items[['order_item_id', 'order_item_name']].drop_duplicates('order_item_id'),
        on='order_item_id', how='left'
    )

    df_final = df_merge2[['product_id', 'order_item_name', 'total_sales']].copy()
    df_final.columns = ['id', 'order_item_name', 'total_sales']
    df_final = df_final.drop_duplicates(subset=['id'], keep='first')

    df_top5 = df_final.sort_values('total_sales', ascending=False).head(5)
    sizes = (df_top5['total_sales'] / df_top5['total_sales'].max()) * 100 + 40

    # Atur posisi bubble
    G = nx.Graph()
    for i in range(len(df_top5)):
        G.add_node(i)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    x_pos = [pos[i][0] for i in range(len(df_top5))]
    y_pos = [pos[i][1] for i in range(len(df_top5))]

    # Build Plotly figure 
    fig_bubble = go.Figure(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        text=[f"{name}<br><b>{int(total):,}</b>" for name, total in zip(df_top5['order_item_name'], df_top5['total_sales'])],
        textposition='middle center',
        textfont=dict(
            color='black',  
            size=12,
            family='Arial'
        ),
        marker=dict(
            size=sizes,
            color=df_top5['total_sales'],
            colorscale='Viridis',
            showscale=True,
            line=dict(width=2, color='DarkSlateGrey'),
            sizemode='diameter',
            opacity=0.7,
        ),
        hoverinfo='text',
        hovertext=[f"{name}<br>Total Sales: {total:,.0f}" for name, total in zip(df_top5['order_item_name'], df_top5['total_sales'])]
    ))

    fig_bubble.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600,
        width=800,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_bubble, use_container_width=True)
    
    # ------------------------------------------------------------------------------------
    # --- Visualisasi Frekuensi Transaksi per Jam berdasarkan IP ---
    st.subheader("Frekuensi Pembelian Berdasarkan Jam dan IP Address", divider=True)

    df_v1 = pd.read_csv('csv/fact_sales_v1.csv', sep=';')
    df_v1 = df_v1[df_v1['order_item_type'] != 'shipping']
    df_v1['order_date'] = pd.to_datetime(df_v1['order_date'])
    df_v1 = df_v1[['order_date', 'ip_address', 'order_id']].drop_duplicates()
    
    # Filter berdasarkan IP dan tanggal
    filtered_df = df_v1[
        (df_v1['ip_address'].isin(selected_ips)) &
        (df_v1['order_date'].dt.date >= date_range[0]) &
        (df_v1['order_date'].dt.date <= date_range[1])
    ]

    # Bulatkan waktu ke jam terdekat
    filtered_df['order_time'] = filtered_df['order_date'].dt.floor('h')

    # Grouping
    grouped = filtered_df.groupby(['order_time', 'ip_address']).size().reset_index(name='jumlah_transaksi')

    # Pivot agar IP Address menjadi kolom
    pivot_df = grouped.pivot(index='order_time', columns='ip_address', values='jumlah_transaksi').fillna(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 8))
    pivot_df.plot(ax=ax, marker='o', markersize=8, linewidth=1.5)
    ax.set_title('Frekuensi Pembelian per Jam Berdasarkan IP Address')
    ax.set_xlabel('Waktu Pemesanan')
    ax.set_ylabel('Jumlah Transaksi')
    ax.legend(title='IP Address', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    plt.tight_layout()

    st.pyplot(fig)
    
    # ------------------------------------------------------------------------------------
    # --- Visualisasi Top 10 Produk dengan Interaksi Terbanyak ---
    st.subheader("Top 10 Produk Berdasarkan Interaksi", divider=True)

    order_items = pd.read_csv("csv/order_items.csv", sep=";")

    # Filter hanya untuk item produk (bukan shipping, tax, dsb)
    product_interactions = order_items[order_items["order_item_type"] == "line_item"]

    # Hitung jumlah interaksi produk
    product_counts = product_interactions["order_item_name"].value_counts().reset_index()
    product_counts.columns = ["product_name", "interaction_count"]

    # Ambil 10 produk teratas
    top_products = product_counts.head(10)

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
    
    
    # ------------------------------------------------------------------------------------
    # --- Visualisasi Sebaran Transaksi per Kota ---
    st.subheader("Sebaran Transaksi per Kota", divider=True)
    st.markdown("Tiap *Point* dilengkapi Jumlah Transaksi dan Produk Terlaris")

    # Tambahkan spinner agar Streamlit tidak tampak diam saat loading lokasi
    with st.spinner("Menyiapkan visualisasi..."):
        city_sales = load_city_sales(selected_ips, date_range)

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


# ---------------------------------- HALAMAN MODELLING ----------------------------------
elif page == "Modelling":
    st.header("🤖 Modelling - Association Rules")
    st.caption("""
Proyek ini menggunakan **mlxtend.frequent_patterns** untuk mengidentifikasi aturan asosiasi dari transaksi e-commerce.
Pengguna dapat memasukkan produk yang dibeli, dan sistem akan merekomendasikan produk lain berdasarkan pola pembelian sebelumnya.
                """)

    # **Input Produk untuk Multi-Select**
    product_list = sorted(set(rules["antecedents"].apply(eval).explode()))  # Mendapatkan daftar produk unik
    selected_products = st.multiselect("Pilih produk yang dibeli:", product_list)

    # **Inferensi Aturan Asosiasi**
    new_transaction = set(selected_products)
    matched_rules = rules[rules["antecedents"].apply(lambda x: set(eval(x)).issubset(new_transaction))]
    
    # bersihkan string set
    def clean_frozenset(text):
        cleaned_text = re.sub(r"frozenset\(\{(.*?)\}\)", r"[\1]", text)  # Ubah ke list format string
        return cleaned_text.replace("'", '"')  # Ganti kutip tunggal ke kutip ganda

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

    # **Visualisasi Confidence dan Lift**
    st.subheader("Visualisasi Confidence dan Lift")
    if not matched_rules.empty:
        fig, ax = plt.subplots()
        sns.barplot(x=matched_rules["consequents"], y=matched_rules["confidence"], color="blue", label="Confidence", ax=ax)
        sns.barplot(x=matched_rules["consequents"], y=matched_rules["lift"], color="red", alpha=0.5, label="Lift", ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.xlabel("Consequents")
        plt.title("Confidence vs Lift dari Produk Rekomendasi")
        plt.legend()
        st.pyplot(fig)
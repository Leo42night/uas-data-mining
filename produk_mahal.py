import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#load dan proses data sales
df = pd.read_csv('csv/fact_sales_v2.csv', sep=';')
print(f"DF: {df.shape[0]} rows x {df.shape[1]} columns")
df.info() # tipe data sesuai (object bisa diubah ke category)
df.head()
df.describe().T # nilai right skewed

print(df.duplicated().sum()) # no duplicate
print(df.isna().sum()) # no missing value

# Baca file dengan delimiter titik koma
df_product = pd.read_csv("csv/product_meta.csv", delimiter=';')
df_sales = pd.read_csv("csv/fact_sales_v2.csv", delimiter=';')

# Bersihkan spasi di nama kolom
df_product.columns = df_product.columns.str.strip()
df_sales.columns = df_sales.columns.str.strip()

# Pastikan kolom product_id sudah sesuai
print("Kolom df_sales:", df_sales.columns.tolist())
print("Kolom df_product:", df_product.columns.tolist())

# Ubah tipe kolom agar bisa digabung
df_sales['product_id'] = df_sales['product_id'].astype(int)
df_product['product_id'] = df_product['product_id'].astype(int)

# Gabungkan kedua DataFrame
df_merged = pd.merge(df_sales, df_product, on='product_id', how='left')
print(df_merged.head())

# Salin dataframe
df_price = df_merged.copy()

# Konversi 'min_price' menjadi float, ganti koma → titik
df_price['min_price'] = pd.to_numeric(
    df_price['min_price'].astype(str).str.replace(',', '.', regex=False),
    errors='coerce'
)

# Ambil 5 produk unik dengan harga tertinggi
top_expensive = (
    df_price[['order_item_name', 'min_price']]
    .dropna()
    .drop_duplicates(subset='order_item_name')
    .sort_values(by='min_price', ascending=False)
    .head(5)
)

# Buat bar chart
plt.figure(figsize=(20, 12))
bars = plt.bar(top_expensive['order_item_name'], top_expensive['min_price'], color='skyblue')
plt.title("Top 5 Produk Termahal yang Pernah Dibeli")
plt.xlabel("Nama Produk")
plt.ylabel("Harga Minimum (Rp)")
plt.xticks(rotation=45, ha='right')

# Format sumbu Y menjadi format Rupiah
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"Rp {x:,.0f}".replace(",", ".")))

plt.tight_layout()
plt.show()

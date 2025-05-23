import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import textwrap

# 1. Load and preprocess sales data
df_sales = pd.read_csv('csv/fact_sales_v2.csv', sep=';')
if df_sales['total_sales'].dtype == object:
  df_sales['total_sales'] = (
      df_sales['total_sales']
      .str.replace('.', '', regex=False)
      .str.replace(',', '.', regex=False)
      .astype(float)
  )
dag_sales = df_sales.groupby('product_id', as_index=False)['total_sales'].sum()

# 2. Load and merge order_product for mapping to order_item_id
df_order_product = pd.read_csv('csv/order_product.csv', sep=';')
df_merge1 = dag_sales.merge(
    df_order_product[['order_item_id', 'product_id']],
    on='product_id', how='left'
)

# 3. Load and merge order_items for product names
df_items = pd.read_csv('csv/order_items.csv', sep=';')
df_merge2 = df_merge1.merge(
    df_items[['order_item_id', 'order_item_name']].drop_duplicates('order_item_id'),
    on='order_item_id', how='left'
)

# 4. Prepare final DataFrame and remove duplicates
result = df_merge2[['product_id', 'order_item_name', 'total_sales']].copy()
result.columns = ['id', 'order_item_name', 'total_sales']
result = result.drop_duplicates(subset=['id'], keep='first')

# 5. Sort all products by sales descending
result_sorted = result.sort_values('total_sales', ascending=False)

# 6. Wrap names and prepare values
wrapped_names = ["\n".join(textwrap.wrap(name, width=15)) for name in result_sorted['order_item_name']]
values = result_sorted['total_sales'].values

# 7. Colormap based on sales values
norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
cmap = cm.viridis
bar_colors = cmap(norm(values))

# 8. Plot horizontal bar chart for all products
total_items = len(wrapped_names)
fig_height = max(6, total_items * 0.3)
plt.figure(figsize=(12, fig_height))
bars = plt.barh(wrapped_names, values, color=bar_colors)
plt.xlabel('Total Penjualan', fontsize=10)
plt.title('Total Penjualan per Produk (Unique product_id)', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)
# Shrink tick label font size to avoid overlap
plt.xticks(fontsize=8)
plt.yticks(fontsize=6)
plt.tight_layout()

# 9. Save chart
graph_path = 'sales_distribution.png'
plt.savefig(graph_path, dpi=300)
print(f"Chart semua produk disimpan sebagai '{graph_path}'")

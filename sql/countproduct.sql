SELECT COUNT(DISTINCT product_id) AS total_produk 
FROM wpkr_wc_product_meta_lookup
WHERE product_id IS NOT NULL AND product_id <> '';

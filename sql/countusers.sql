SELECT COUNT(*) AS total_email_customer
FROM wpkr_wc_customer_lookup
WHERE email IS NOT NULL AND email <> '';

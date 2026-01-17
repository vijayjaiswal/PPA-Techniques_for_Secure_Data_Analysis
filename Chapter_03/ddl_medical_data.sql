CREATE TABLE customer_data (
    record_id INT,
    full_name TEXT,
    email TEXT,
    phone_number TEXT,
    national_id TEXT,
    age INT,
    gender CHAR(1),
    zip_code TEXT,
    city TEXT,
    occupation TEXT,
    medical_condition TEXT,
    annual_income INT
);

truncate customer_data;

insert into customer_data values(1,'Rahul Sharma','rahul.sharma@gmail.com',919877000000,'IDA1234567',27,'M',110001,'Delhi','Software Engineer','Diabetes',1800000);
insert into customer_data values(2,'Anita Verma','anita.verma@yahoo.com',919812000000,'IDB2345678',29,'F',110001,'Delhi','Data Analyst','Asthma',1200000);
insert into customer_data values(3,'Suresh Iyer','suresh.iyer@outlook.com',919823000000,'IDC3456789',25,'M',110001,'Bengaluru','Project Manager','Hypertension',2200000);
insert into customer_data values(4,'Neha Gupta','neha.gupta@gmail.com',919835000000,'IDD4567890',28,'F',110001,'Bengaluru','Product Manager','Diabetes',2000000);
insert into customer_data values(5,'Amit Patel','amit.patel@gmail.com',919846000000,'IDE5678901',52,'M',380001,'Ahmedabad','Business Owner','Cardiac',3000000);
insert into customer_data values(6,'Kavita Rao','kavita.rao@yahoo.com',919857000000,'IDF6789012',47,'F',380001,'Ahmedabad','Consultant','Arthritis',1600000);
insert into customer_data values(7,'Rohan Mehta','rohan.mehta@gmail.com',919868000000,'IDG7890123',51,'M',380001,'Mumbai','Marketing Exec','Anxiety',900000);
insert into customer_data values(8,'Priya Nair','priya.nair@outlook.com',919879000000,'IDH8901234',50,'F',380001,'Mumbai','UX Designer','Migraine',1400000);
insert into customer_data values(9,'Rahul Sharma','rahul.sharma@gmail.com',919871000000,'IDA1234568',27,'M',110001,'Delhi','Software Engineer','Diabetes',1489286);
insert into customer_data values(10,'Anita Verma','anita.verma@yahoo.com',919876000000,'IDB2345679',29,'F',110001,'Delhi','Data Analyst','Asthma',1428571);
insert into customer_data values(11,'Suresh Iyer','suresh.iyer@outlook.com',919881000000,'IDC3456790',25,'M',110001,'Bengaluru','Project Manager','Hypertension',1367857);
insert into customer_data values(12,'Neha Gupta','neha.gupta@gmail.com',919886000000,'IDD4567891',28,'F',110001,'Bengaluru','Product Manager','Diabetes',1307143);
insert into customer_data values(13,'Amit Patel','amit.patel@gmail.com',919891000000,'IDE5678902',52,'M',380001,'Ahmedabad','Business Owner','Cardiac',1246429);
insert into customer_data values(14,'Kavita Rao','kavita.rao@yahoo.com',919895000000,'IDF6789013',47,'F',380001,'Ahmedabad','Consultant','Arthritis',1185714);
insert into customer_data values(15,'Rohan Mehta','rohan.mehta@gmail.com',919900000000,'IDG7890124',51,'M',380001,'Mumbai','Marketing Exec','Anxiety',1125000);
insert into customer_data values(16,'Priya Nair','priya.nair@outlook.com',919905000000,'IDH8901235',50,'F',380001,'Mumbai','UX Designer','Migraine',1064286);
insert into customer_data values(17,'Rahul Sharma','rahul.sharma@gmail.com',919910000000,'IDA1234569',27,'M',110001,'Delhi','Software Engineer','Diabetes',1003571);
insert into customer_data values(18,'Anita Verma','anita.verma@yahoo.com',919915000000,'IDB2345680',29,'F',110001,'Delhi','Data Analyst','Asthma',942857);
insert into customer_data values(19,'Suresh Iyer','suresh.iyer@outlook.com',919919000000,'IDC3456791',25,'M',110001,'Bengaluru','Project Manager','Hypertension',882143);
insert into customer_data values(20,'Neha Gupta','neha.gupta@gmail.com',919924000000,'IDD4567892',28,'F',110001,'Bengaluru','Product Manager','Diabetes',821429);
insert into customer_data values(21,'Amit Patel','amit.patel@gmail.com',919929000000,'IDE5678903',52,'M',380001,'Ahmedabad','Business Owner','Cardiac',760714);
insert into customer_data values(22,'Kavita Rao','kavita.rao@yahoo.com',919934000000,'IDF6789014',47,'F',380001,'Ahmedabad','Consultant','Arthritis',700000);
insert into customer_data values(23,'Rohan Mehta','rohan.mehta@gmail.com',919939000000,'IDG7890125',51,'M',380001,'Mumbai','Marketing Exec','Anxiety',639286);
insert into customer_data values(24,'Priya Nair','priya.nair@outlook.com',919944000000,'IDH8901236',50,'F',380001,'Mumbai','UX Designer','Migraine',578571);
insert into customer_data values(25,'Rahul Sharma','rahul.sharma@gmail.com',919948000000,'IDA1234570',27,'M',110001,'Delhi','Software Engineer','Diabetes',517857);
insert into customer_data values(26,'Anita Verma','anita.verma@yahoo.com',919953000000,'IDB2345681',29,'F',110001,'Delhi','Data Analyst','Asthma',457143);
insert into customer_data values(27,'Suresh Iyer','suresh.iyer@outlook.com',919958000000,'IDC3456792',25,'M',110001,'Bengaluru','Project Manager','Hypertension',396429);
insert into customer_data values(28,'Neha Gupta','neha.gupta@gmail.com',919963000000,'IDD4567893',28,'F',110001,'Bengaluru','Product Manager','Diabetes',335714);
insert into customer_data values(29,'Amit Patel','amit.patel@gmail.com',919968000000,'IDE5678904',52,'M',380001,'Ahmedabad','Business Owner','Cardiac',275000);
insert into customer_data values(30,'Kavita Rao','kavita.rao@yahoo.com',919973000000,'IDF6789015',47,'F',380001,'Ahmedabad','Consultant','Arthritis',214286);


SELECT record_id,LEFT(email, 2) || '****@' || SPLIT_PART(email, '@', 2) AS email_masked FROM customer_data;
SELECT record_id, 'XXXXXX' || RIGHT(phone_number, 4) AS phone_masked FROM customer_data;
SELECT record_id, LEFT(national_id, 2) || '*****' || RIGHT(national_id, 2) AS national_id_masked FROM customer_data;
SELECT record_id, CASE
	WHEN annual_income < 1500000 THEN '< 15L'
	WHEN annual_income BETWEEN 1500000 AND 2000000 THEN '15L–20L'
		ELSE '> 20L'
	END AS income_band
FROM customer_data;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

SELECT record_id, encode(digest(email, 'sha256'), 'hex') AS email_hash FROM customer_data;

SELECT record_id, encode(digest(national_id || 'SALT_2026', 'sha256'), 'hex') AS national_id_hash FROM customer_data;

SELECT record_id,
	LEFT(full_name, 1) || '****' AS name_masked,
	LEFT(email, 2) || '****@' || SPLIT_PART(email, '@', 2) AS email_masked,
	encode(digest(national_id, 'sha256'), 'hex') AS national_id_hash,
	zip_code, city, medical_condition FROM customer_data;

SELECT
	record_id,
    CASE
        WHEN current_user = 'admin' THEN full_name
        ELSE LEFT(full_name, 1) || '****'
	    END AS full_name,
	CASE
		WHEN current_user = 'admin' THEN email
		ELSE LEFT(email, 2) || '****@' || SPLIT_PART(email, '@', 2)
	END AS email
	FROM customer_data;


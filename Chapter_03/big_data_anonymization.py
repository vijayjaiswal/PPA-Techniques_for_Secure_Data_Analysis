import os
import sys

# Configure environment for Spark
# Unset SPARK_HOME to ensure we use the PySpark bundled jars instead of system installation
# This fixes the "getPythonAuthSocketTimeout does not exist in the JVM" error caused by version mismatch.
if 'SPARK_HOME' in os.environ:
    del os.environ['SPARK_HOME']

# Set HADOOP_HOME to local directory for Windows compatibility (winutils.exe)
current_dir = os.path.dirname(os.path.abspath(__file__))
hadoop_home = os.path.join(current_dir, "hadoop_home")
if os.path.exists(hadoop_home):
    os.environ['HADOOP_HOME'] = hadoop_home
    os.environ["HADOOP_OPTS"] = "-Djava.library.path="
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ['PATH'] += os.pathsep + os.path.join(hadoop_home, "bin")

# Set JAVA_HOME to JDK 17 (required for Spark 4.0)
os.environ['JAVA_HOME'] = r"D:\installed\Java\jdk-17"

# Ensure Spark uses the same Python executable as the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, floor, concat, substring, count, countDistinct, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def create_spark_session():
    return SparkSession.builder \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.sql.warehouse.dir", "file:/tmp/spark-warehouse") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "1") \
        .appName("BigDataAnonymization") \
        .master("local[*]") \
        .getOrCreate()

def load_data(spark):
    """
    Loads data from 'medical_data.csv' with the expanded schema.
    """
    # Define the schema strictly to ensure types (int vs string) are correct
    schema = StructType([
        StructField("record_id", IntegerType(), True),
        StructField("full_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("phone_number", StringType(), True),
        StructField("national_id", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("zip_code", StringType(), True),
        StructField("city", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("medical_condition", StringType(), True),
        StructField("annual_income", IntegerType(), True)
    ])
    
    # Read from CSV file
    # We use the schema defined above to enforce types immediately upon load
    return spark.read.csv(os.path.join(current_dir, "medical_data.csv"), header=True, schema=schema)

def generalize_data(df):
    """
    Applies generalization logic to Quasi-Identifiers (QIs).
    """
    print("\n--- Applying Generalization ---")
    
    # 1. Generalize zip_code: Keep first 3 digits, mask the rest (e.g., 13053 -> 130**)
    # 2. Generalize age: Bucket into 10-year intervals (e.g., 28 -> 20-29)
    # 3. We retain 'gender' as a QI without generalization for this example.
    
    return df.withColumn("zip_code_gen", 
            concat(substring(col("zip_code"), 1, 3), lit("**"))
        ).withColumn("age_gen", 
            concat(
                (floor(col("age") / 10) * 10).cast("string"), 
                lit("-"), 
                ((floor(col("age") / 10) * 10) + 9).cast("string")
            )
        )

def check_k_anonymity(df, k):
    """
    Verifies k-anonymity by grouping by QIs (zip_code_gen, age_gen, gender).
    Filters out groups smaller than k (Suppression).
    """
    print(f"\n--- Checking k-Anonymity (k={k}) ---")
    
    # Group by the Generalized QIs. Note: We added 'gender' to the grouping.
    # If we include 'gender', groups become smaller and harder to satisfy k.
    group_cols = ["zip_code_gen", "age_gen", "gender"]
    
    group_counts = df.groupBy(group_cols).agg(count("*").alias("group_size"))
    
    # Join back to original data to flag records
    analyzed_df = df.join(group_counts, on=group_cols, how="left")
    
    # Identify compliant records
    compliant_df = analyzed_df.filter(col("group_size") >= k)
    suppressed_df = analyzed_df.filter(col("group_size") < k)
    
    print(f"Total Records: {df.count()}")
    print(f"Compliant Records: {compliant_df.count()}")
    print(f"Suppressed Records: {suppressed_df.count()}")
    
    if suppressed_df.count() > 0:
        print("Sample Suppressed Records:")
        suppressed_df.select(*group_cols, "medical_condition", "group_size").show(5)
        
    return compliant_df

def check_l_diversity(df, l):
    """
    Verifies l-diversity on the k-anonymous data.
    Ensures each group has at least 'l' distinct sensitive values (medical_condition).
    """
    print(f"\n--- Checking l-diversity (l={l}) ---")
    
    # QIs used for grouping
    group_cols = ["zip_code_gen", "age_gen", "gender"]
    
    # Calculate distinct sensitive values per group
    diversity_counts = df.groupBy(group_cols) \
        .agg(countDistinct("medical_condition").alias("distinct_conditions"))
        
    # Join back
    analyzed_df = df.join(diversity_counts, on=group_cols, how="left")
    
    # Filter
    compliant_df = analyzed_df.filter(col("distinct_conditions") >= l)
    non_compliant_df = analyzed_df.filter(col("distinct_conditions") < l)
    
    print(f"k-Anonymous Records: {df.count()}")
    print(f"l-Diverse Records: {compliant_df.count()}")
    
    if non_compliant_df.count() > 0:
        print("Records failing l-diversity (Risk of Homogeneity Attack):")
        non_compliant_df.select(*group_cols, "medical_condition", "distinct_conditions").show(5)
        
    return compliant_df

if __name__ == "__main__":
    try:
        spark = create_spark_session()
    
        # 1. Load Raw Data
        raw_df = load_data(spark)
        raw_df.show()
    
        # 2. Generalize (Transform QIs)
        # We drop Direct Identifiers (full_name, email, etc) immediately.
        # We select Generalized QIs + Sensitive Attributes.
        anonymized_candidates = generalize_data(raw_df) \
            .select("zip_code_gen", "age_gen", "gender", "medical_condition", "annual_income")
        
        # 3. Enforce k-Anonymity (k=3)
        # We use k=2 here because adding 'gender' splits our small data significantly.
        k_anonymous_data = check_k_anonymity(anonymized_candidates, k=2)
    
        # 4. Enforce l-diversity (l=2)
        final_secure_data = check_l_diversity(k_anonymous_data, l=2)
    
        print("\n--- Final Secure Dataset ---")
        final_secure_data.orderBy("zip_code_gen", "age_gen", "gender").show()

        # Export to CSV
        print("\n--- Writing output to file ---")
        #final_secure_data.coalesce(1).write.csv("anonymized_output.csv", header=True, mode="overwrite")
        # Check for hadoop.dll on Windows to decide write method
        hadoop_dll_exists = False
        if os.name == 'nt' and 'HADOOP_HOME' in os.environ:
             hadoop_dll_exists = os.path.exists(os.path.join(os.environ['HADOOP_HOME'], 'bin', 'hadoop.dll'))

        if hadoop_dll_exists:
            try:
                final_secure_data.write.mode("overwrite").csv("output/anonymized_data")
                print("Output written to 'output/anonymized_data' folder via Spark.")
            except Exception as e:
                print(f"Spark write failed ({e}). Falling back to Pandas.")
                final_secure_data.toPandas().to_csv("anonymized_output.csv", index=False)
                print("Output written to 'anonymized_output.csv' via Pandas.")
        else:
            print("Note: hadoop.dll not found in HADOOP_HOME/bin. Skipping Spark write to avoid Windows NativeIO errors.")
            final_secure_data.toPandas().to_csv("anonymized_output.csv", index=False)
            print("Output written to 'anonymized_output.csv' via Pandas.")
    
        spark.stop()
    except Exception as e:
        import traceback
        print(f"Error executing Spark job: {e}")
        traceback.print_exc()
        exit(1)
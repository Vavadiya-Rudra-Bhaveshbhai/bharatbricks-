from pyspark.sql import functions as F

train_table = "workspace.default.train_00000_of_00001"
eval_table = "workspace.default.eval_00000_of_00001"
bronze_table = "workspace.default.bronze_rbi_circular_qa"

train_df = (
    spark.table(train_table)
    .withColumn("split", F.lit("train"))
    .withColumn("source_table", F.lit(train_table))
)

eval_df = (
    spark.table(eval_table)
    .withColumn("split", F.lit("eval"))
    .withColumn("source_table", F.lit(eval_table))
)

bronze_df = (
    train_df.unionByName(eval_df, allowMissingColumns=True)
    .withColumn("ingested_at", F.current_timestamp())
)

(
    bronze_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)

display(spark.table(bronze_table).limit(20))
print("Bronze table created:", bronze_table)
print("Bronze rows:", spark.table(bronze_table).count())

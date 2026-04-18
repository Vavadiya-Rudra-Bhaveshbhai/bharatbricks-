from pyspark.sql import functions as F

bronze_table = "workspace.default.bronze_rbi_circular_qa"
silver_table = "workspace.default.silver_rbi_circular_qa"
gold_chunks_table = "workspace.default.gold_rbi_circular_chunks"
gold_eval_table = "workspace.default.gold_rbi_eval"

raw_df = spark.table(bronze_table)

clean_df = (
    raw_df
    .withColumn("document", F.trim(F.col("document")))
    .withColumn("filename", F.trim(F.col("filename")))
    .withColumn("regulation_area", F.trim(F.col("regulation_area")))
    .withColumn("applicable_to", F.trim(F.col("applicable_to")))
    .withColumn("issued_on", F.expr("try_cast(issued_on as date)"))
    .withColumn("chunks_text", F.regexp_replace(F.col("chunks_text"), r"\s+", " "))
    .withColumn("question", F.regexp_replace(F.col("question"), r"\s+", " "))
    .withColumn("answer", F.regexp_replace(F.col("answer"), r"\s+", " "))
    .withColumn(
        "qa_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("split"), F.lit("")),
                F.coalesce(F.col("document"), F.lit("")),
                F.coalesce(F.col("filename"), F.lit("")),
                F.coalesce(F.col("question"), F.lit("")),
                F.coalesce(F.col("answer"), F.lit("")),
            ),
            256,
        ),
    )
)

(
    clean_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)

chunks_df = (
    clean_df
    .select(
        "document",
        "filename",
        "regulation_area",
        "applicable_to",
        "issued_on",
        "key_topics",
        "chunks_text",
        "is_table",
    )
    .dropna(subset=["document", "filename", "chunks_text"])
    .dropDuplicates(["document", "filename", "chunks_text"])
    .withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("document"), F.lit("")),
                F.coalesce(F.col("filename"), F.lit("")),
                F.coalesce(F.col("chunks_text"), F.lit("")),
            ),
            256,
        ),
    )
    .withColumn(
        "retrieval_text",
        F.concat_ws(
            "\n",
            F.concat(F.lit("Document: "), F.coalesce(F.col("document"), F.lit(""))),
            F.concat(F.lit("Filename: "), F.coalesce(F.col("filename"), F.lit(""))),
            F.concat(F.lit("Regulation area: "), F.coalesce(F.col("regulation_area"), F.lit(""))),
            F.concat(F.lit("Applicable to: "), F.coalesce(F.col("applicable_to"), F.lit(""))),
            F.concat(F.lit("Issued on: "), F.coalesce(F.col("issued_on").cast("string"), F.lit(""))),
            F.concat(F.lit("Chunk: "), F.coalesce(F.col("chunks_text"), F.lit(""))),
        ),
    )
)

(
    chunks_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_chunks_table)
)

eval_df = (
    clean_df
    .filter(F.col("split") == "eval")
    .select(
        "qa_id",
        "document",
        "question",
        "answer",
        "rephrased_question",
        "rephrased_answer",
        "evaluation_criteria",
        "category",
        "estimated_difficulty",
    )
)

(
    eval_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_eval_table)
)

print("Silver rows:", spark.table(silver_table).count())
print("Gold chunk rows:", spark.table(gold_chunks_table).count())
print("Gold eval rows:", spark.table(gold_eval_table).count())

display(spark.table(gold_chunks_table).limit(20))

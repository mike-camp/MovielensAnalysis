package com.mikecamp.movielens

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.types
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.functions
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SQLContext


object Predictor {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("movielens")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val sqlContext = new SQLContext(sc)
    val movies = loadMovies(sc,sqlContext)
    val user = loadUsers(sqlContext)
    val data = loadUserItemMatrix(sqlContext)
    movies.registerTempTable("movies")
    user.registerTempTable("users")
    data.registerTempTable("data")
    val dataWithReviews = sqlContext.sql("""
        WITH a AS (SELECT * FROM data as a LEFT JOIN users as b
          ON a.userID=b.id)
        SELECT a.userID, a.itemID, a.rating, a.occupation, 
        a.gender , c.genres, c.year
        FROM a LEFT JOIN movies as c ON a.itemID=c.id
      """
      )
        
//    dataWithReviews.show()
//    dataWithReviews.printSchema()

    val pipeline = createPipeline()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(
          new RegressionEvaluator()
            .setMetricName("rmse")
            .setLabelCol("rating")
            .setPredictionCol("lrPreds"))
      .setNumFolds(3)
      .setEstimatorParamMaps(new ParamGridBuilder().build())
    val Array(df1,df2) = dataWithReviews.randomSplit(Array(.5,.5))
    df2.show()
    val debuggingPipeline = createDebuggingPipeline()
    val fittedPipeline = debuggingPipeline.fit(df1)
    val predictions = fittedPipeline.transform(df2)
    predictions.show()

    val evaluator = new RegressionEvaluator()
            .setMetricName("rmse")
            .setLabelCol("rating")
            .setPredictionCol("lrPreds")
    println(evaluator.evaluate(predictions))
    
//    val cvModel = cv.fit(dataWithReviews)
//    for (i <-cvModel.avgMetrics) {
//      println(i)
//    }
    
  }
  

  def createDebuggingPipeline() = {
    val als = new ALS()
      .setUserCol("userID")
      .setItemCol("itemID")
      .setRatingCol("rating")
      .setPredictionCol("alsPreds")
      .setRank(10)
      .setNonnegative(true)
      .setAlpha(.1)
    val meanFiller = new MeanFiller()
    //val exprs = numDf.columns.map(c => coalesce(col(c), col(s"avg($c)"), lit(0.0)).alias(c))
    val occupationIndexer = new StringIndexer()
      .setInputCol("occupation")
      .setOutputCol("indexedOccupations")
    val genderIndexer = new StringIndexer()
      .setInputCol("gender")
      .setOutputCol("indexedGender")

    val genderEncoder = new OneHotEncoder()
      .setInputCol("indexedGender")
      .setOutputCol("oneHotGender")
    val oneHotEncoder = new OneHotEncoder()
      .setInputCol("indexedOccupations")
      .setOutputCol("occupationVectors")
    val cols = Array("occupationVectors","genres","predictionsNaNRemoved",
          "oneHotGender","year")
    val vecAssemble = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("finalVec")
      
    val linearRegression = new LinearRegression()
      .setFeaturesCol(vecAssemble.getOutputCol)
      .setLabelCol("rating")
      .setPredictionCol("lrPreds")
      
    new Pipeline()
      .setStages(Array(als,meanFiller,
          occupationIndexer,genderIndexer,genderEncoder,
          oneHotEncoder, vecAssemble, linearRegression))
    
  }
  
  def createPipeline() = {
    val als = new ALS()
      .setUserCol("userID")
      .setItemCol("itemID")
      .setRatingCol("rating")
      .setPredictionCol("alsPreds")
      .setRank(10)
      .setNonnegative(true)
      .setAlpha(.1)
    val meanFiller = new MeanFiller()
    //val exprs = numDf.columns.map(c => coalesce(col(c), col(s"avg($c)"), lit(0.0)).alias(c))
    val occupationIndexer = new StringIndexer()
      .setInputCol("occupation")
      .setOutputCol("indexedOccupations")
    val genderIndexer = new StringIndexer()
      .setInputCol("gender")
      .setOutputCol("indexedGender")

    val genderEncoder = new OneHotEncoder()
      .setInputCol("indexedGender")
      .setOutputCol("oneHotGender")
    val oneHotEncoder = new OneHotEncoder()
      .setInputCol("indexedOccupations")
      .setOutputCol("occupationVectors")
    val cols = Array("occupationVectors","genres","predictionsNaNRemoved",
          "oneHotGender","year")
    val vecAssemble = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("finalVec")
      
    val linearRegression = new LinearRegression()
      .setFeaturesCol(vecAssemble.getOutputCol)
      .setLabelCol("rating")
      .setPredictionCol("lrPreds")
      
    new Pipeline()
      .setStages(Array(als,meanFiller,
          occupationIndexer,genderIndexer,genderEncoder,
          oneHotEncoder, vecAssemble, linearRegression))
    
  }
  
  def loadMovies(sc:SparkContext, sqlContext:SQLContext):DataFrame = {
    val textFile = sc.textFile("ml-100k/u.item")
    val rowFile = textFile.map( row => {
      val lines = row.split("""\|""")
      val genres = lines.slice(5, 24).map( _ match {
        case s:String => s.toDouble
        case _ => 0.0
      } )
      val id = lines(0).toInt
      val title = lines(1)
      val date = lines(2).split("-")
      val year = date(date.length-1)
      val intYear = if (year.length==4) year.toInt else 1999
      Row(id, title, intYear, Vectors.dense(genres))
    })
    val schema = types.StructType(Seq(
        types.StructField("id",types.IntegerType),
        types.StructField("name",types.StringType),
        types.StructField("year",types.IntegerType),
        types.StructField("genres",VectorType)
//        types.StructField("genres",types.ArrayType(types.DoubleType))
        ))
    sqlContext.createDataFrame(rowFile, schema)
  }
  def loadUsers(sqlContext:SQLContext):DataFrame = {
    val schema = types.StructType(Seq(
      types.StructField("id",types.IntegerType),
      types.StructField("age",types.IntegerType),
      types.StructField("gender",types.StringType),
      types.StructField("occupation",types.StringType),
      types.StructField("zipcode",types.StringType)
      ))
    sqlContext.read.format("csv")
      .option("delimiter","|")
      .option("header",false)
      .schema(schema)
      .load("ml-100k/u.user")
  }
  def loadUserItemMatrix(sqlContext:SQLContext):DataFrame = {
    val schema = types.StructType(Seq(
        types.StructField("userID",types.IntegerType),
        types.StructField("itemID",types.IntegerType),
        types.StructField("rating",types.DoubleType),
        types.StructField("date",types.LongType)
        ))
    sqlContext.read.format("csv")
      .option("delimiter","\t")
      .option("header",false)
      .schema(schema)
      .load("ml-100k/u.data")
      
  }
}
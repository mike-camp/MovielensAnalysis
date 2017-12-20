package com.mikecamp.movielens
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{DataType, DataTypes}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{StructType,StructField,DoubleType}
import org.apache.spark.sql.{DataFrame,Dataset}
import org.apache.spark.ml.Model

class MeanFiller(override val uid: String) 
  extends Model[MeanFiller] {
   def this() = this(Identifiable.randomUID("genreTransformer"))
//  override def copy(extra: ParamMap)  = {
//    defaultCopy(extra)
//  }
 override def copy(extra: ParamMap) = {
     new MeanFiller()
   }
  
  override def transformSchema(schema: StructType): StructType = {
    schema.add(StructField("predictionsNaNRemoved",DoubleType))
  }
  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.registerTempTable("temp")
    //a.userID, a.itemID, a.rating, a.occupation, a.gender , b.genre

    dataset.sqlContext.sql("""
      SELECT userID, itemID, rating, occupation, 
        gender, genres, year, alsPreds,
      COALESCE(alsPreds,AVG(rating) OVER (PARTITION BY itemID), 
        AVG(rating) OVER ())  AS predictionsNaNRemoved FROM temp""" 
      ).na.drop()
  }
}
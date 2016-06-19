import org.apache.hadoop.conf.Configuration
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import java.util.logging.Logger
import org.apache.spark.sql.types._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors.dense
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.sql.Row
import org.bson.BSONObject
import com.mongodb.hadoop.MongoInputFormat
import org.apache.spark.sql.types.StringType


object MongoSpark extends App {

  //Setup the logger. This just prints to STDOUT for now
  val logger = Logger.getLogger(MongoSpark.getClass.getName())
  logger.info("Testing out my lizznogger")

  //Configure the spark session and mongoDB connection
  val mongoConfig = new Configuration()
  mongoConfig.set("mongo.input.uri", "mongodb://cabe:carolina1@localhost:27017/RAS.income")
  val sparkConf = new SparkConf()
  val sc = new SparkContext("local", "MongoSpark", sparkConf)
  val sql = new SQLContext(sc)

  //Create an RDD from mongoDB connection
  val documentsRaw = sc.newAPIHadoopRDD(
    mongoConfig,
    classOf[MongoInputFormat],
    classOf[Object],
    classOf[BSONObject]
  )

  //mongoDB schema definition
  val schema = (new StructType)
    .add("id", (new StructType).add("$oid", StringType))
    .add("age", DoubleType)
    .add("capital_gain", DoubleType)
    .add("capital_loss", DoubleType)
    .add("education", StringType)
    .add("educational_num", DoubleType)
    .add("fnlwgt", DoubleType)
    .add("gender", StringType)
    .add("hours_per_week", DoubleType)
    .add("income", StringType)
    .add("marital_status", StringType)
    .add("occupation", StringType)
    .add("race", StringType)
    .add("relationship", StringType)
    .add("workclass", StringType)

  //ETL work to convert the BSON documents from MongoDB to a usable format for the
  //mllib algorithms to work with.
  val documents = documentsRaw.map(x => x._2) //Grab just the BSONObject objects
  val documentsJSON = documents.map(x => x.toString()) //Convert to flattened JSON
  val df = sql.read.schema(schema).json(documentsJSON) //Convert to dataframe using predefined schema
  val dfSkinny = df.select("age", "income", "hours_per_week") //Grab only the regression columns
  val regPoints = dfSkinny.map(r => labelize(r)) //Transform dataframe to RDD[LabeledPoints]

  //Normalize the vector of features so that the regression will preform better
  val scaler1 = new StandardScaler().fit(regPoints.map(x => x.features))
  val regRDD = regPoints.map(r => (r.label, scaler1.transform(r.features)))
  val regLabel = regRDD.map(r =>  new LabeledPoint(r._1, r._2))

  //Run the logistic regression
  val model = new LogisticRegressionWithLBFGS()
    .run(regLabel)
  val modelStr = model.weights

  println(modelStr) //Print out the model weights

  //Function to transform RDD Row to a LabeledPoint object. This is required by the LogisticRegression Algorithm
  def labelize (row: Row): LabeledPoint = {
    val label = row.getString(1) match {
      case "<=50K" => 1.0
      case ">50K" => 0.0
    }
    val features = new Array[Double](2)
    features(0) = row.getDouble(0)
    features(1) = row.getDouble(2)
    val featuresVec: Vector = dense(features)
    return LabeledPoint(label, featuresVec)
  }


}
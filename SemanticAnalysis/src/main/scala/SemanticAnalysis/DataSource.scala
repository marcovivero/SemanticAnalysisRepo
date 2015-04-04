package SemanticAnalysis

import grizzled.slf4j.Logger
import io.prediction.controller._
import io.prediction.data.storage.Storage
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {

    // Get PEvents database instance.
    val eventsDb = Storage.getPEvents()

    // Pull sentences as a SentArray which consists of a sentiment attribute
    // which records the observation's sentiment class value as a double, and
    // a phrase attribute which will be later transformed as an NGram Vector.
    val phrases = eventsDb.find(
      appId = dsp.appId,
      entityType = Some("source"),
      eventNames = Some(List("phrases"))

    // Convert collected events to SentArray objects.
    )(sc).map(e => SentArray(
      e.properties.get[Double]("sentiment"),
      e.properties.get[String]("phrase")
    ))


    new TrainingData(phrases)
  }
}

case class SentArray(
                       sentiment : Double,
                       phrase : String
                      ) extends Serializable

class TrainingData(
                    val labeledPhrases: RDD[SentArray]
                    ) extends Serializable

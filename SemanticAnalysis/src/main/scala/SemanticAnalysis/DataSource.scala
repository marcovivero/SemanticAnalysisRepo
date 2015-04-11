package SemanticAnalysis

import grizzled.slf4j.Logger
import io.prediction.controller._
import io.prediction.data.storage.Storage
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

case class DataSourceParams(
                             appId: Int,
                             evalK: Option[Int]
                             ) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  def readEventData(sc: SparkContext) : RDD[SentArray] = {
    // Get PEvents database instance.
    val eventsDb = Storage.getPEvents()

    // Pull sentences as a SentArray which consists of a sentiment attribute
    // which records the observation's sentiment class value as a double, and
    // a phrase attribute which will be later transformed as an NGram Vector.
    eventsDb.find(
      appId = dsp.appId,
      entityType = Some("source"),
      eventNames = Some(List("phrases"))

      // Convert collected events to SentArray objects.
    )(sc).map(e => SentArray(
      e.properties.get[Double]("sentiment"),
      e.properties.get[String]("phrase")
    ))
  }

  override
  def readTraining(sc: SparkContext): TrainingData = {
    new TrainingData(readEventData(sc))
  }

  override
  def readEval(sc : SparkContext) :
    Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, ActualResult)])] = {
    require(!dsp.evalK.isEmpty, "DataSourceParams.evalK must not be None")

    // Read event data.
    val labeledPhrases : RDD[SentArray] = readEventData(sc)

    // Split k-folds.
    val evalK = dsp.evalK.get
    val indexedPhrases : RDD[(SentArray, Long)] = labeledPhrases.zipWithIndex

    (0 until evalK).map {e =>
      val trainingPoints = indexedPhrases.filter(f => f._2 % evalK != e).map(f => f._1)
      val testPoints = indexedPhrases.filter(f => f._2 % evalK == e).map(f => f._1)

      (
        new TrainingData(trainingPoints),
        new EmptyEvaluationInfo(),
        testPoints.map {
          e => (new Query(e.phrase), new ActualResult(e.sentiment))
        }
        )
    }
  }
}



case class SentArray(
                       sentiment : Double,
                       phrase : String
                      ) extends Serializable

class TrainingData(
                    val labeledPhrases: RDD[SentArray]
                    ) extends Serializable

package SemanticAnalysis

import java.util.HashSet

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import SemanticAnalysis.NGramMachine

case class AlgorithmParams(
                          nGramWindow : Int,
                          lambda : Double
                            ) extends Params

class Algorithm(val params : AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {


  def train (data : PreparedData) : Model = {
    val newData = data.labeledPhrases
      .map(e => DummyData(
      e.sentiment,
      NGramMachine.extract(e.phrase, params.nGramWindow)
    ))
    val dataIter = newData.toLocalIterator
    NaiveBayes.train(data.labeledPoints, params.lambda)
  }

  def predict (model: NaiveBayesModel, query: Query) : PredictedResult = {
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }
}

case class DummyData (
                       sentiment : Double,
                       phrase : HashSet[String]
                       ) extends Serializable

class Model () extends Serializable


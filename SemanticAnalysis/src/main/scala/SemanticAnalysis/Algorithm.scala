package SemanticAnalysis

import java.util.HashMap

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.JavaConverters._
import scala.collection.mutable

case class AlgorithmParams(
                          nGramWindow : Int,
                          lambda : Double
                            ) extends Params

class Algorithm(val params : AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {


  def train (data : PreparedData) : Model = {

    // Create training data universe of n-grams.
    val NGramUniverse = NGramMachine.create_universe(
      data.labeledPhrases
        .map(e => DummyData(
        NGramMachine.extract(
          e.phrase, params.nGramWindow
        )).nGrams
        ).toLocalIterator.asJava
    )


    NaiveBayes.train(labeledPoints, params.lambda)
  }

  def predict (model: NaiveBayesModel, query: Query) : PredictedResult = {
    val label = model.predict(Vectors.dense(query.phrase))
    new PredictedResult(label)
  }
}

case class DummyData (
                       nGrams : HashMap[String, Integer]
                       ) extends Serializable


class Model () extends Serializable


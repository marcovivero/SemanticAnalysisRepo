package SemanticAnalysis

import java.util.HashMap

import SemanticAnalysis.NGramMachine.{create_universe, extract, hash2Vect}
import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.JavaConverters._

case class AlgorithmParams(
                          nGramWindow : Int,
                          lambda : Double
                            ) extends Params

class Algorithm(val params : AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {


  def train (data : PreparedData) : Model = {

    // Create training data universe of n-grams.
    val NGramUniverse = create_universe(
      data.labeledPhrases
        .map(e => DummyData(
            extract(e.phrase, params.nGramWindow)
          ).nGrams
        ).toLocalIterator.asJava
    )

    val transformedData = data.labeledPhrases.map(e => TransformedData(
        LabeledPoint(
          e.sentiment,
          Vectors.dense(
            hash2Vect(
              extract(e.phrase, params.nGramWindow),
              NGramUniverse)
          ))))

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

case class TransformedData(
                          labeledPoint: LabeledPoint
                            )


class Model () extends Serializable


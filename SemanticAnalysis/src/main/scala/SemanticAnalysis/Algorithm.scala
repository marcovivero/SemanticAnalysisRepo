package SemanticAnalysis

import java.util.LinkedHashSet
import java.util.HashMap

import SemanticAnalysis.NGramMachine.{create_universe, extract, hash2Vect}
import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.JavaConverters._
import scala.collection.mutable

case class AlgorithmParams(
                          nGramWindow : Int,
                          lambda : Double
                            ) extends Params

class Algorithm(val params : AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {


  def train (data : PreparedData) : Model = {

    val NGramUniverse = create_universe(
      data.labeledPhrases
        .map(e => DummyData(
        extract(e.phrase, params.nGramWindow)
      ).nGrams
        ).toLocalIterator.asJava
    )
    // Create training data universe of n-grams.

    val transformedData = data.labeledPhrases.map(e => LabeledPoint(
          e.sentiment,
          Vectors.dense(
            hash2Vect(
              extract(e.phrase, params.nGramWindow),
              NGramUniverse)
          )))

    new Model(
      NGramUniverse,
      NaiveBayes.train(transformedData, params.lambda)
    )

  }

  def predict (model : Model, query: Query) : PredictedResult = {
    val label = model.nb.predict(
      Vectors.dense(
        hash2Vect(
          extract(query.phrase, params.nGramWindow),
          model.universe
      )))
    new PredictedResult(label)
  }
}

case class DummyData (
                       nGrams : HashMap[String, Integer]
                       ) extends Serializable


case class Model (universe : LinkedHashSet[String],
             nb : NaiveBayesModel) extends Serializable


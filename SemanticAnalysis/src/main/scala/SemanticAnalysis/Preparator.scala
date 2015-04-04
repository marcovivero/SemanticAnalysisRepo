/**
 * Created by Marco on 4/2/15.
 */

package SemanticAnalysis

import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


class PreparedData (
                     val labeledPhrases : RDD[SentArray]
                     ) extends Serializable

class Preparator extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc : SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(trainingData.labeledPhrases)
  }


}
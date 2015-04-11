package SemanticAnalysis

import io.prediction.controller._
import scala.math.pow

case class MeanSquaredError() extends AverageMetric[EmptyEvaluationInfo, Query, PredictedResult, ActualResult] {
  def calculate(query: Query, predicted: PredictedResult, actual: ActualResult)
  : Double = pow(predicted.sentiment - actual.sentiment, 2)
}

object meanSquaredErrorEvaluation extends Evaluation {
  // Define Engine and Metric used in Evaluation
  engineMetric = (SemanticAnalysisEngine(), new MeanSquaredError())
}

object EngineParamsList extends EngineParamsGenerator {
  // Define list of EngineParams used in Evaluation

  // First, we define the base engine params. It specifies the appId from which
  // the data is read, and a evalK parameter is used to define the
  // cross-validation.
  private[this] val baseEP = EngineParams(
    dataSourceParams = DataSourceParams(appId = 19, evalK = Some(5)))

  // Second, we specify the engine params list by explicitly listing all
  // algorithm parameters. In this case, we evaluate 3 engine params, each with
  // a different algorithm params value.

  // In this example we will primarily focus on the appropriate value of the additive smoothing constant,
  // and leave the number of n-grams fixed. The number of n-grams itself is a model hyperparameter
  // and should also be tuned.
  engineParamsList = Seq(
    baseEP.copy(algorithmParamsList = Seq(("algo", AlgorithmParams(2, 10.0)))),
    baseEP.copy(algorithmParamsList = Seq(("algo", AlgorithmParams(2, 100.0)))),
    baseEP.copy(algorithmParamsList = Seq(("algo", AlgorithmParams(2, 1000.0)))))
}
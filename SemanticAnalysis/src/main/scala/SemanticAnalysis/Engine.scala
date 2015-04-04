package SemanticAnalysis

import io.prediction.controller.{Engine, IEngineFactory}

class Query(
             val features: Array[Double]
             ) extends Serializable

class PredictedResult(
                       val label: Double
                       ) extends Serializable

object SemanticAnalysisEngine extends IEngineFactory {
  override
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("algo" -> classOf[Algorithm]),
      classOf[Serving])
  }
}

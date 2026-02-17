package oracle

import (
	"math"
	"testing"
)

func TestTrainAndForecast(t *testing.T) {
	series := make([]float64, 0, 80)
	for i := 0; i < 80; i++ {
		series = append(series, 5+0.8*float64(i))
	}

	cfg := TrainConfig{
		Lag:          6,
		Hidden:       10,
		Epochs:       1400,
		LearningRate: 0.006,
		Seed:         7,
	}

	result, err := Train(series, cfg)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	predictions, err := Forecast(result, series, 3)
	if err != nil {
		t.Fatalf("forecast failed: %v", err)
	}

	if len(predictions) != 3 {
		t.Fatalf("predictions len = %d, want 3", len(predictions))
	}
	if math.IsNaN(predictions[0]) || math.IsInf(predictions[0], 0) {
		t.Fatalf("prediction is not finite: %v", predictions[0])
	}
	if predictions[0] <= series[len(series)-1]-5 {
		t.Fatalf("unexpected first prediction: got %.3f", predictions[0])
	}
}

func TestValidate(t *testing.T) {
	series := make([]float64, 0, 100)
	for i := 0; i < 100; i++ {
		series = append(series, 20+0.6*float64(i))
	}

	cfg := TrainConfig{
		Lag:          8,
		Hidden:       12,
		Epochs:       1800,
		LearningRate: 0.007,
		Seed:         11,
	}

	trainSeries := series[:90]
	result, err := Train(trainSeries, cfg)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	metrics, err := Validate(result, series, 10)
	if err != nil {
		t.Fatalf("validate failed: %v", err)
	}

	if metrics.Count != 10 {
		t.Fatalf("count = %d, want 10", metrics.Count)
	}
	if math.IsNaN(metrics.MAE) || math.IsInf(metrics.MAE, 0) {
		t.Fatalf("MAE is not finite: %v", metrics.MAE)
	}
	if math.IsNaN(metrics.RMSE) || math.IsInf(metrics.RMSE, 0) {
		t.Fatalf("RMSE is not finite: %v", metrics.RMSE)
	}
	if math.IsNaN(metrics.MAPE) || math.IsInf(metrics.MAPE, 0) {
		t.Fatalf("MAPE is not finite: %v", metrics.MAPE)
	}
	if metrics.MAE > 5 {
		t.Fatalf("MAE too large: %.4f", metrics.MAE)
	}
}

func TestValidateRejectsInvalidInput(t *testing.T) {
	series := make([]float64, 0, 30)
	for i := 0; i < 30; i++ {
		series = append(series, 1+float64(i))
	}

	result, err := Train(series[:24], TrainConfig{
		Lag:          6,
		Hidden:       10,
		Epochs:       1000,
		LearningRate: 0.008,
		Seed:         3,
	})
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	if _, err := Validate(result, series, 0); err == nil {
		t.Fatalf("expected error for zero holdout")
	}
	if _, err := Validate(result, series, len(series)); err == nil {
		t.Fatalf("expected error for oversized holdout")
	}
}

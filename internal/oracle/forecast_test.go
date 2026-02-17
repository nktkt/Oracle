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

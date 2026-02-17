package oracle

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadModelRoundTrip(t *testing.T) {
	series := make([]float64, 0, 90)
	for i := 0; i < 90; i++ {
		series = append(series, 15+0.5*float64(i))
	}

	result, err := Train(series, TrainConfig{
		Lag:          7,
		Hidden:       11,
		Epochs:       1600,
		LearningRate: 0.007,
		Seed:         9,
	})
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	expected, err := Forecast(result, series, 4)
	if err != nil {
		t.Fatalf("forecast (expected) failed: %v", err)
	}

	path := filepath.Join(t.TempDir(), "oracle_model.json")
	if err := SaveModel(path, result); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if loaded.Lag != result.Lag {
		t.Fatalf("loaded lag = %d, want %d", loaded.Lag, result.Lag)
	}
	if math.Abs(loaded.Scaler.Mean-result.Scaler.Mean) > 1e-12 {
		t.Fatalf("scaler mean mismatch: got %v, want %v", loaded.Scaler.Mean, result.Scaler.Mean)
	}

	actual, err := Forecast(loaded, series, 4)
	if err != nil {
		t.Fatalf("forecast (loaded) failed: %v", err)
	}

	if len(actual) != len(expected) {
		t.Fatalf("len(actual) = %d, want %d", len(actual), len(expected))
	}
	for i := range expected {
		if math.Abs(actual[i]-expected[i]) > 1e-9 {
			t.Fatalf("prediction[%d] mismatch: got %.12f, want %.12f", i, actual[i], expected[i])
		}
	}
}

func TestSaveModelRejectsNil(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.json")
	if err := SaveModel(path, nil); err == nil {
		t.Fatalf("expected SaveModel error for nil result")
	}
}

func TestLoadModelRejectsInvalidParameters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad_model.json")
	body := `{"version":1,"lag":3,"scaler":{"Mean":0,"Std":1},"w1":[[1,2,3],[1,2]],"b1":[0,0],"w2":[0,0],"b2":0}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	if _, err := LoadModel(path); err == nil {
		t.Fatalf("expected LoadModel error for invalid shape")
	}
}

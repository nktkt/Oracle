package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildForecastPoints(t *testing.T) {
	points := buildForecastPoints([]float64{10, 11.5}, 0.5)
	if len(points) != 2 {
		t.Fatalf("len(points) = %d, want 2", len(points))
	}
	if points[0].Step != 1 || points[1].Step != 2 {
		t.Fatalf("unexpected steps: %+v", points)
	}
	if points[0].Prediction != 10 {
		t.Fatalf("unexpected prediction: %+v", points[0])
	}
}

func TestWriteForecastCSV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "forecast.csv")
	points := []ForecastPoint{
		{Step: 1, Prediction: 12.3, Low95: 11.8, High95: 12.8},
		{Step: 2, Prediction: 13.1, Low95: 12.6, High95: 13.6},
	}

	if err := writeForecastCSV(path, points); err != nil {
		t.Fatalf("writeForecastCSV failed: %v", err)
	}

	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	text := string(body)
	if !strings.Contains(text, "step,prediction,low_95,high_95") {
		t.Fatalf("missing header: %s", text)
	}
	if !strings.Contains(text, "1,12.300000,11.800000,12.800000") {
		t.Fatalf("missing first row: %s", text)
	}
}

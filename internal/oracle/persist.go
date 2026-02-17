package oracle

import (
	"encoding/json"
	"fmt"
	"os"
)

const modelFormatVersion = 1

type persistedModel struct {
	Version        int          `json:"version"`
	Lag            int          `json:"lag"`
	Scaler         Standardizer `json:"scaler"`
	MSE            float64      `json:"mse"`
	ResidualStdDev float64      `json:"residual_std_dev"`
	W1             [][]float64  `json:"w1"`
	B1             []float64    `json:"b1"`
	W2             []float64    `json:"w2"`
	B2             float64      `json:"b2"`
}

func SaveModel(path string, result *TrainResult) error {
	if result == nil || result.Model == nil {
		return fmt.Errorf("invalid train result")
	}

	pm := persistedModel{
		Version:        modelFormatVersion,
		Lag:            result.Lag,
		Scaler:         result.Scaler,
		MSE:            result.MSE,
		ResidualStdDev: result.ResidualStdDev,
		W1:             result.Model.W1,
		B1:             result.Model.B1,
		W2:             result.Model.W2,
		B2:             result.Model.B2,
	}

	if err := validatePersistedModel(pm); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	enc := json.NewEncoder(file)
	enc.SetIndent("", "  ")
	if err := enc.Encode(pm); err != nil {
		return err
	}
	return nil
}

func LoadModel(path string) (*TrainResult, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var pm persistedModel
	dec := json.NewDecoder(file)
	if err := dec.Decode(&pm); err != nil {
		return nil, err
	}

	if err := validatePersistedModel(pm); err != nil {
		return nil, err
	}

	model := &MLP{
		InputSize:  pm.Lag,
		HiddenSize: len(pm.B1),
		W1:         clone2D(pm.W1),
		B1:         append([]float64(nil), pm.B1...),
		W2:         append([]float64(nil), pm.W2...),
		B2:         pm.B2,
	}

	return &TrainResult{
		Model:          model,
		Scaler:         pm.Scaler,
		Lag:            pm.Lag,
		MSE:            pm.MSE,
		ResidualStdDev: pm.ResidualStdDev,
	}, nil
}

func validatePersistedModel(pm persistedModel) error {
	if pm.Version != modelFormatVersion {
		return fmt.Errorf("unsupported model version: %d", pm.Version)
	}
	if pm.Lag <= 0 {
		return fmt.Errorf("invalid lag in model: %d", pm.Lag)
	}
	if len(pm.W1) == 0 || len(pm.B1) == 0 || len(pm.W2) == 0 {
		return fmt.Errorf("empty model parameters")
	}
	if len(pm.W1) != len(pm.B1) || len(pm.B1) != len(pm.W2) {
		return fmt.Errorf("hidden layer parameter size mismatch")
	}
	for i, row := range pm.W1 {
		if len(row) != pm.Lag {
			return fmt.Errorf("w1[%d] width mismatch: got %d, want %d", i, len(row), pm.Lag)
		}
	}
	if pm.Scaler.Std <= 0 {
		return fmt.Errorf("invalid scaler std: %f", pm.Scaler.Std)
	}
	return nil
}

func clone2D(src [][]float64) [][]float64 {
	out := make([][]float64, len(src))
	for i := range src {
		out[i] = append([]float64(nil), src[i]...)
	}
	return out
}

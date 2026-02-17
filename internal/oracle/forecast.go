package oracle

import (
	"fmt"
	"math"
	"math/rand"
)

type TrainConfig struct {
	Lag          int
	Hidden       int
	Epochs       int
	LearningRate float64
	Seed         int64
}

type TrainResult struct {
	Model          *MLP
	Scaler         Standardizer
	Lag            int
	MSE            float64
	ResidualStdDev float64
}

func Train(series []float64, cfg TrainConfig) (*TrainResult, error) {
	if len(series) < 6 {
		return nil, fmt.Errorf("series too short: need at least 6 points")
	}

	if cfg.Lag <= 0 {
		cfg.Lag = 6
	}
	if cfg.Hidden <= 0 {
		cfg.Hidden = 12
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1800
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.008
	}
	if len(series) <= cfg.Lag {
		return nil, fmt.Errorf("series length must be larger than lag")
	}

	scaler := Standardizer{}
	scaler.Fit(series)
	normalized := scaler.TransformSlice(series)
	x, y := makeWindows(normalized, cfg.Lag)
	if len(x) == 0 {
		return nil, fmt.Errorf("failed to build training windows")
	}

	rnd := rand.New(rand.NewSource(cfg.Seed))
	model := NewMLP(cfg.Lag, cfg.Hidden, rnd)
	order := make([]int, len(x))
	for i := range order {
		order[i] = i
	}

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		rnd.Shuffle(len(order), func(i, j int) {
			order[i], order[j] = order[j], order[i]
		})

		for _, idx := range order {
			in := x[idx]
			target := y[idx]

			h, out, err := model.Forward(in)
			if err != nil {
				return nil, err
			}

			diff := out - target
			dOut := 2.0 * diff

			dZ1 := make([]float64, cfg.Hidden)
			for j := 0; j < cfg.Hidden; j++ {
				dZ1[j] = dOut * model.W2[j] * (1 - h[j]*h[j])
			}

			for j := 0; j < cfg.Hidden; j++ {
				model.W2[j] -= cfg.LearningRate * dOut * h[j]
			}
			model.B2 -= cfg.LearningRate * dOut

			for j := 0; j < cfg.Hidden; j++ {
				for i := 0; i < cfg.Lag; i++ {
					model.W1[j][i] -= cfg.LearningRate * dZ1[j] * in[i]
				}
				model.B1[j] -= cfg.LearningRate * dZ1[j]
			}
		}
	}

	mse, stdDev, err := evaluate(model, scaler, series, x, cfg.Lag)
	if err != nil {
		return nil, err
	}

	return &TrainResult{
		Model:          model,
		Scaler:         scaler,
		Lag:            cfg.Lag,
		MSE:            mse,
		ResidualStdDev: stdDev,
	}, nil
}

func Forecast(result *TrainResult, observed []float64, steps int) ([]float64, error) {
	if result == nil || result.Model == nil {
		return nil, fmt.Errorf("invalid train result")
	}
	if len(observed) < result.Lag {
		return nil, fmt.Errorf("observed series shorter than lag")
	}
	if steps <= 0 {
		return []float64{}, nil
	}

	history := append([]float64(nil), observed...)
	predictions := make([]float64, 0, steps)

	for i := 0; i < steps; i++ {
		start := len(history) - result.Lag
		window := history[start:]
		normalizedWindow := result.Scaler.TransformSlice(window)

		nextNorm, err := result.Model.Predict(normalizedWindow)
		if err != nil {
			return nil, err
		}

		next := result.Scaler.Inverse(nextNorm)
		predictions = append(predictions, next)
		history = append(history, next)
	}

	return predictions, nil
}

func makeWindows(series []float64, lag int) ([][]float64, []float64) {
	count := len(series) - lag
	x := make([][]float64, 0, count)
	y := make([]float64, 0, count)

	for i := lag; i < len(series); i++ {
		window := make([]float64, lag)
		copy(window, series[i-lag:i])
		x = append(x, window)
		y = append(y, series[i])
	}

	return x, y
}

func evaluate(model *MLP, scaler Standardizer, rawSeries []float64, windows [][]float64, lag int) (float64, float64, error) {
	if len(windows) == 0 {
		return 0, 0, fmt.Errorf("no evaluation windows")
	}

	sumSq := 0.0
	residuals := make([]float64, 0, len(windows))

	for i, w := range windows {
		predNorm, err := model.Predict(w)
		if err != nil {
			return 0, 0, err
		}

		pred := scaler.Inverse(predNorm)
		actual := rawSeries[lag+i]
		r := actual - pred
		residuals = append(residuals, r)
		sumSq += r * r
	}

	mse := sumSq / float64(len(residuals))
	meanResidual := 0.0
	for _, r := range residuals {
		meanResidual += r
	}
	meanResidual /= float64(len(residuals))

	variance := 0.0
	for _, r := range residuals {
		d := r - meanResidual
		variance += d * d
	}
	variance /= float64(len(residuals))
	stdDev := math.Sqrt(variance)

	return mse, stdDev, nil
}

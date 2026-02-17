package oracle

import (
	"fmt"
	"math"
	"math/rand"
)

type Standardizer struct {
	Mean float64
	Std  float64
}

func (s *Standardizer) Fit(values []float64) {
	if len(values) == 0 {
		s.Mean = 0
		s.Std = 1
		return
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	s.Mean = sum / float64(len(values))

	variance := 0.0
	for _, v := range values {
		d := v - s.Mean
		variance += d * d
	}

	s.Std = math.Sqrt(variance / float64(len(values)))
	if s.Std < 1e-9 {
		s.Std = 1
	}
}

func (s Standardizer) Transform(v float64) float64 {
	return (v - s.Mean) / s.Std
}

func (s Standardizer) Inverse(v float64) float64 {
	return v*s.Std + s.Mean
}

func (s Standardizer) TransformSlice(values []float64) []float64 {
	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = s.Transform(v)
	}
	return out
}

type MLP struct {
	InputSize  int
	HiddenSize int
	W1         [][]float64
	B1         []float64
	W2         []float64
	B2         float64
}

func NewMLP(inputSize, hiddenSize int, rnd *rand.Rand) *MLP {
	m := &MLP{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		W1:         make([][]float64, hiddenSize),
		B1:         make([]float64, hiddenSize),
		W2:         make([]float64, hiddenSize),
	}

	scale := 1.0 / math.Sqrt(float64(inputSize))
	for j := 0; j < hiddenSize; j++ {
		m.W1[j] = make([]float64, inputSize)
		for i := 0; i < inputSize; i++ {
			m.W1[j][i] = (rnd.Float64()*2 - 1) * scale
		}
		m.W2[j] = (rnd.Float64()*2 - 1) * scale
	}
	return m
}

func (m *MLP) Forward(x []float64) ([]float64, float64, error) {
	if len(x) != m.InputSize {
		return nil, 0, fmt.Errorf("input size mismatch: got %d, want %d", len(x), m.InputSize)
	}

	h := make([]float64, m.HiddenSize)
	for j := 0; j < m.HiddenSize; j++ {
		sum := m.B1[j]
		for i := 0; i < m.InputSize; i++ {
			sum += m.W1[j][i] * x[i]
		}
		h[j] = math.Tanh(sum)
	}

	out := m.B2
	for j := 0; j < m.HiddenSize; j++ {
		out += m.W2[j] * h[j]
	}

	return h, out, nil
}

func (m *MLP) Predict(x []float64) (float64, error) {
	_, out, err := m.Forward(x)
	return out, err
}

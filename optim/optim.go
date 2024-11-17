package optim

import (
	"iter"
	"lberg/tinyGoGrad/nn"

	"golang.org/x/exp/constraints"
)

type Optimiser[F constraints.Float] interface {
	ZeroGrad()
	Parameters() iter.Seq[*nn.Parameter[F]]
	AddParameter(*nn.Parameter[F])
	Step()
}

type BaseOptimiser[F constraints.Float] struct {
	parameters []*nn.Parameter[F]
}

func (s *BaseOptimiser[F]) ZeroGrad() {
	for _, p := range s.parameters {
		p.Grad = 0
	}
}

func (s *BaseOptimiser[F]) AddParameter(param *nn.Parameter[F]) {
	s.parameters = append(s.parameters, param)
}

func (s *BaseOptimiser[F]) Parameters() iter.Seq[*nn.Parameter[F]] {
	return func(yield func(*nn.Parameter[F]) bool) {
		for _, p := range s.parameters {
			if !yield(p) {
				return
			}
		}
	}
}

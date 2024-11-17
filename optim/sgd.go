package optim

import (
	"golang.org/x/exp/constraints"
)

type SGD[F constraints.Float] struct {
	BaseOptimiser[F]
	lr F
}

func NewSGD[F constraints.Float](lr F) *SGD[F] {
	return &SGD[F]{lr: lr}
}

func (s *SGD[F]) Step() {
	for p := range s.Parameters() {
		p.Val = p.Val - s.lr*p.Grad
	}
}

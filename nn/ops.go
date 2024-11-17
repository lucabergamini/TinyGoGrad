package nn

import (
	"math"

	"golang.org/x/exp/constraints"
)

func Exp[F constraints.Float](par *Parameter[F]) *Parameter[F] {
	val := F(math.Exp(float64(par.Val)))
	return &Parameter[F]{
		Val: val,
		ins: []*inputRef[F]{{
			param:      par,
			derivative: val,
		}},
	}
}

func Log2[F constraints.Float](par *Parameter[F]) *Parameter[F] {
	return &Parameter[F]{
		Val: F(math.Log2(float64(par.Val))),
		ins: []*inputRef[F]{{
			param:      par,
			derivative: 1.0 / par.Val,
		}},
	}
}

func Pow[F constraints.Float](par *Parameter[F], at float64) *Parameter[F] {
	return &Parameter[F]{
		Val: F(math.Pow(float64(par.Val), at)),
		ins: []*inputRef[F]{{
			param:      par,
			derivative: F(at * math.Pow(float64(par.Val), at-1)),
		}},
	}
}

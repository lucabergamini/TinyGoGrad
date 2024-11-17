package nn

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type Parameter[F constraints.Float] struct {
	Val  F
	Grad F
	ins  []*inputRef[F]
}

type inputRef[F constraints.Float] struct {
	param      *Parameter[F]
	derivative F
}

func NewParameter[F constraints.Float](value F) *Parameter[F] {
	return &Parameter[F]{Val: value}
}

func (p *Parameter[F]) Neg() *Parameter[F] {
	return &Parameter[F]{
		Val: -p.Val,
		ins: []*inputRef[F]{{
			param:      p,
			derivative: -1,
		}},
	}
}

func (p *Parameter[F]) Rec() *Parameter[F] {
	return &Parameter[F]{
		Val: 1 / p.Val,
		ins: []*inputRef[F]{{
			param:      p,
			derivative: -1 / (p.Val * p.Val),
		}},
	}
}

func (p *Parameter[F]) Plus(oth any) *Parameter[F] {

	withBaseType := func(oth F) *Parameter[F] {
		newParam := Parameter[F]{Val: p.Val + oth}
		newParam.ins = []*inputRef[F]{{
			param:      p,
			derivative: 1,
		}}
		return &newParam
	}

	switch oth := oth.(type) {
	case int:
		return withBaseType(F(oth))
	case float32:
		return withBaseType(F(oth))
	case float64:
		return withBaseType(F(oth))
	case *Parameter[F]:
		newParam := Parameter[F]{Val: p.Val + oth.Val}
		newParam.ins = []*inputRef[F]{{
			param:      p,
			derivative: 1,
		}, {
			param:      oth,
			derivative: 1,
		}}
		return &newParam

	}
	panic(fmt.Sprintf("plus not implemented for %T", oth))
}

func (p *Parameter[F]) Minus(oth any) *Parameter[F] {
	switch oth := oth.(type) {
	case int:
		return p.Plus(-oth)
	case float32:
		return p.Plus(-oth)
	case float64:
		return p.Plus(-oth)
	case *Parameter[F]:
		return p.Plus(oth.Neg())
	}
	panic(fmt.Sprintf("minus not implemented for %T", oth))
}

func (p *Parameter[F]) Mul(oth any) *Parameter[F] {
	withBaseType := func(oth F) *Parameter[F] {
		newParam := Parameter[F]{Val: p.Val * oth}
		newParam.ins = []*inputRef[F]{{
			param:      p,
			derivative: F(oth),
		}}
		return &newParam

	}
	switch oth := oth.(type) {
	case int:
		return withBaseType(F(oth))
	case float32:
		return withBaseType(F(oth))
	case float64:
		return withBaseType(F(oth))
	case *Parameter[F]:
		newParam := Parameter[F]{Val: p.Val * oth.Val}
		newParam.ins = []*inputRef[F]{{
			param:      p,
			derivative: oth.Val,
		}, {
			param:      oth,
			derivative: p.Val,
		}}
		return &newParam
	}
	panic(fmt.Sprintf("mul not implemented for %T", oth))
}

func (p *Parameter[F]) Div(oth any) *Parameter[F] {
	switch oth := oth.(type) {
	case int:
		return p.Mul(1.0 / oth)
	case float32:
		return p.Mul(1 / oth)
	case float64:
		return p.Mul(1 / oth)

	case *Parameter[F]:
		return p.Mul(oth.Rec())
	}
	panic(fmt.Sprintf("div not implemented for %T", oth))
}

func (p *Parameter[F]) Backward() {
	var inner func(p *Parameter[F], partDer F)
	inner = func(p *Parameter[F], partDer F) {
		// NOTE(@lberg): gradient is summed from different branches
		p.Grad += partDer
		for _, in := range p.ins {
			// NOTE(@lberg): chain rule
			inner(in.param, in.derivative*partDer)
		}
	}
	inner(p, 1)
}

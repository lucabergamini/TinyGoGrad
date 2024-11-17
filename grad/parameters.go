package grad

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type Parameter[F constraints.Float] struct {
	Val  F
	Grad F
	ins  []*ChainRef[F]
}

type ChainRef[F constraints.Float] struct {
	param      *Parameter[F]
	derivative F
}

func NewInputParameter[F constraints.Float](value F) *Parameter[F] {
	return &Parameter[F]{Val: value}
}

func (p *Parameter[F]) Neg() *Parameter[F] {
	return &Parameter[F]{
		Val: -p.Val,
		ins: []*ChainRef[F]{{
			param:      p,
			derivative: -1,
		}},
	}
}

func (p *Parameter[F]) Rec() *Parameter[F] {
	return &Parameter[F]{
		Val: 1 / p.Val,
		ins: []*ChainRef[F]{{
			param:      p,
			derivative: -1 / (p.Val * p.Val),
		}},
	}
}

func (p *Parameter[F]) Plus(oth any) *Parameter[F] {

	plusBase := func(oth F) *Parameter[F] {
		newParam := Parameter[F]{Val: p.Val + oth}
		newParam.ins = []*ChainRef[F]{{
			param:      p,
			derivative: 1,
		}}
		return &newParam
	}

	switch oth := oth.(type) {
	case int:
		return plusBase(F(oth))
	case float32:
		return plusBase(F(oth))
	case float64:
		return plusBase(F(oth))
	case *Parameter[F]:
		newParam := Parameter[F]{Val: p.Val + oth.Val}
		newParam.ins = []*ChainRef[F]{{
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
	mulBase := func(oth F) *Parameter[F] {
		newParam := Parameter[F]{Val: p.Val * oth}
		newParam.ins = []*ChainRef[F]{{
			param:      p,
			derivative: F(oth),
		}}
		return &newParam

	}
	switch oth := oth.(type) {
	case int:
		return mulBase(F(oth))
	case float32:
		return mulBase(F(oth))
	case float64:
		return mulBase(F(oth))
	case *Parameter[F]:
		newParam := Parameter[F]{Val: p.Val * oth.Val}
		newParam.ins = []*ChainRef[F]{{
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
	inner = func(p *Parameter[F], partDev F) {
		// NOTE(@lberg): gradient is summed from different branches
		p.Grad += partDev
		for _, in := range p.ins {
			// NOTE(@lberg): chain rule
			inner(in.param, in.derivative*partDev)
		}
	}
	inner(p, 1)
}

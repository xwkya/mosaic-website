import React, { Component } from 'react';
import CustomSlider from './Slider.js'

// Select tile number slider
class TilesSelector extends Component {
    constructor(props) {
        super(props)
    }
    state = {
        value: 32
    }
    tilesMarks = [
        {
            value: 16,
            label: '16',
        },
        {
            value: 20,
        },
        {
            value: 24,
            label: '24',
        },
        {
            value: 28,
        },
        {
            value: 32,
            label: '32',
        },
        {
            value: 36,
        },
        {
            value: 40,
            label: '40',
        },
        {
            value: 44,
        },
        {
            value: 48,
            label: '48',
            color: '#fff'
        }
    ]

    handleChange = (event, value) => {
        this.setState({value: value})
        this.props.onParameterChange(event)
    }

    render() {
        return (
            <CustomSlider
                        min={16}
                        step={4}
                        max={48}
                        style ={{
                            marginLeft: '5%',
                            marginRight: '5%',
                        }}
                        marks = {this.tilesMarks}
                        onChange = {this.handleChange}
                        defaultValue={32}
                    />
        )
    }
}

export default TilesSelector
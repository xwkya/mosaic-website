import React, { Component } from 'react';
import CustomSlider from './Slider.js'

// Select improvement value slider
class ImprovementSelector extends Component {
    constructor(props) {
        super(props)
    }
    
    state = {
        value: 0.5
    }

    tilesMarks = [
        {
            value: 0,
            label: '0',
        },
        {
            value: 0.25,
        },
        {
            value: 0.5,
            label: '0.5',
        },
        {
            value: 0.75,
        },
        {
            value: 1,
            label: '1',
        }
    ]

    handleChange = (event, value) => {
        this.setState({value: value})
        this.props.onParameterChange(event)
    }

    render() {
        return (
            <CustomSlider
                        min={0}
                        step={0.05}
                        max={1}
                        style = {{
                            marginLeft: '5%',
                            marginRight: '5%',
                        }}
                        marks = {this.tilesMarks}
                        defaultValue={0.5}
                        onChange={this.handleChange}
                    />
        )
    }
}

export default ImprovementSelector
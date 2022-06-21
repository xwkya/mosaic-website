import React, { Component } from 'react';
import CustomSlider from './Slider.js'

// Select improvement value slider
class ImprovementSelector extends Component {
    constructor(props) {
        super(props)
    }
    
    state = {
        value: 0.25
    }

    tilesMarks = [
        {
            value: 0,
            label: '0',
        },
        {
            value: 0.1,
            label: '0.1',
        },
        {
            value: 0.2,
            label: '0.2',
        },
        {
            value: 0.3,
            label: '0.3',
        },
        {
            value: 0.4,
            label: '0.4',
        },
        {
            value: 0.5,
            label: '0.5',
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
                        max={0.5}
                        style = {{
                            marginLeft: '5%',
                            marginRight: '5%',
                        }}
                        marks = {this.tilesMarks}
                        defaultValue={0.25}
                        onChange={this.handleChange}
                    />
        )
    }
}

export default ImprovementSelector
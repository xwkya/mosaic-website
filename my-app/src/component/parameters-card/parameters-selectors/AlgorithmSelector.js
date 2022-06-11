import React, { Component } from 'react';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import BootstrapInput from './Select.js'
import MenuItem from '@mui/material/MenuItem';

// Select algorithn box
class AlgorithmSelector extends Component {
    constructor(props){
        super(props)
    }

    state = {algorithm: 'Cosine'}

    actionOnChange = event => {
        this.setState({algorithm: event.target.value});
        this.props.onParameterChange(event)
    }


    render () {
        return (
            <FormControl sx={{ m: 1, minWidth: 120 }} style={{width: '90%'}}>
                <Select
                    labelId="alg-select"
                    id="alg-select"
                    value={this.state.algorithm}
                    defaultValue={'Cosine'}
                    label="Algorithm"
                    onChange={this.actionOnChange}
                    style ={{
                        color: '#ffffff',
                        width: '90%'
                    }}
                    sx={{borderColor: '#ffffff'}}
                    input={<BootstrapInput />}

                    >
                    <MenuItem value={'Network'}>Network</MenuItem>
                    <MenuItem value={'Cosine'}>Cosine</MenuItem>
                    <MenuItem value={'Average'}>Average</MenuItem>
                </Select>
            </FormControl>
        )
    }
}

export default AlgorithmSelector;
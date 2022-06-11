import React, { Component } from 'react';
import Button from '@mui/material/Button';
import LoadingButton from '@mui/lab/LoadingButton';
import { CircularProgress } from '@mui/material';
import ImprovementSelector from './parameters-selectors/ImprovementSelector';
import AlgorithmSelector from './parameters-selectors/AlgorithmSelector';
import TilesSelector from './parameters-selectors/TilesSelector';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardActions from '@mui/material/CardActions';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import { Stack } from '@mui/material';

class ParametersCard extends Component {
    constructor(props) {
        super(props)

    }

    state = {
        improvement: 0.5,
        algorithm: 'Cosine',
        tiles: 32,
    }

    improvementChange = event => {
        this.props.onImprovementChange(event)
        this.setState({improvement: event.target.value})
    }

    algorithmChange = event => {
        this.props.onAlgorithmChange(event)
        this.setState({algorithm: event.target.value})
    }

    tilesChange = event => {
        this.props.onTilesChange(event)
        this.setState({tiles: event.target.value})
    }
    loadingIndic = () => {
        return <CircularProgress color="success" />
    }

    uploadClick = (event) => {
        this.props.onFileUpload(event)
    }
    

    uploadButton = () => {
        return(
            <LoadingButton
            onClick={this.uploadClick}
            loading={!this.props.uploadClickable}
            variant="contained"
            color="secondary"
            loadingIndicator= {<CircularProgress color="white" />}
            disabled={!this.props.uploadClickable}
            style= {{width: '80%', height:'45px'}}
            >
            Upload
            </LoadingButton>
        )
    }

    tilesSelector = <TilesSelector onParameterChange={this.tilesChange}/>
    improvementSelector = <ImprovementSelector onParameterChange={this.improvementChange}/>
    algorithmSelector = <AlgorithmSelector onParameterChange= {this.algorithmChange}/>

    render () {
        if (this.props.display) {
            return (
                <Card sx={{ width: '100%', marginTop: '15px'}}>
                    <Stack spacing={-3}>
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 20, color: '#ffffff'}} align='left' gutterBottom>
                                Algorithm parameters
                            </Typography>
                        </CardContent>
                        
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 14, color: '#ffffff'}} align='left' gutterBottom>
                                Number of tiles
                            </Typography>
                        </CardContent>
                        <CardActions className= 'Card'>
                            {this.tilesSelector}
                        </CardActions>

                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 14, color: '#ffffff'}} align='left' gutterBottom>
                                Improvement Ratio
                            </Typography>
                        </CardContent>
                        <CardActions className= 'Card'>
                            {this.improvementSelector}
                        </CardActions>
                        
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 14, color: '#ffffff'}} align='left' gutterBottom>
                                Strategy
                            </Typography>
                        </CardContent>
                        
                        <CardActions className= 'Card'>
                                <Grid container spacing={2}
                                    alignItems="center"
                                    justifyContent="center">
                                    <Grid item xs={6}>
                                        {this.algorithmSelector}
                                    </Grid>
                                    <Grid item xs={6}>
                                        {this.uploadButton()}
                                    </Grid>
                                </Grid>
                        </CardActions>
                    </Stack>
                </Card>
            )
        }
        else {
            return
        }
    }
}

export default ParametersCard;
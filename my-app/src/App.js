import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import Button from '@mui/material/Button';
import React, { Component } from 'react';
import CollectionsIcon from '@mui/icons-material/Collections';
import ParametersCard from './component/parameters-card/ParametersCard.js'
import MosaicDisplay from './component/mosaic-display/MosaicDisplay.js'
import Grid from '@mui/material/Grid';
import FileData from './component/parameters-card/FileData';
import { Box, Container } from '@mui/material';
import AboutMeCard from './component/about-me/AboutMeCard';

class App extends Component {

	state = {
		// Initially, no file is selected
		selectedFile: null,
		returnedImage: null,
        hasUploadedImage: false,
        algorithm: 'Cosine',
        tiles: 32,
        improvement: 0.5

	};

    tilesChange = (event) => {
        this.setState({tiles: event.target.value})
    }

    algorithmChange = (event) => {
        this.setState({algorithm: event.target.value})
    }

    improvementChange = (event) => {
        this.setState({improvement: event.target.value})
    }

	// On file select (from the pop up)
	onFileChange = event => {

		// Update the state
		this.setState({ selectedFile: event.target.files[0] });
	};

	// On file upload (click the upload button)
	onFileUpload = () => {
        // The image is sent to the server
        this.setState({hasUploadedImage: true})

		// Create an object of formData
		const formData = new FormData();

		// Update the formData object
		formData.append(
			"file",
			this.state.selectedFile,
			this.state.selectedFile.name
		);
        formData.append('algorithm', this.state.algorithm)
        formData.append('tiles', this.state.tiles)
        formData.append('improvement', this.state.improvement)

		// Request made to the backend api
		// Send formData object
		axios({
			method: 'post',
			url: 'http://127.0.0.1:5000/uploader',
			data: formData,
			responseType: 'blob'
		}).then((response) => {
			console.log(response);
			this.setState({returnedImage: response.data, hasUploadedImage: false});
		  }, (error) => {
			console.log(error);
		  });
	};

    //mosaicDisplayInst = this.mosaicDisplay({returnedImage: this.state.returnedImage, onFileUpload: this.onFileUpload, parent: this})

	render() {
		return (
			<div className="App">
                <Grid container
                    spacing={4}
                    justifyContent="center"
                    style={{width: '100%'}}>

                    
                    <Grid item xs= {3}>
                        <Container>
                            <FileData selectedFile={this.state.selectedFile}/>
                            <ParametersCard
                                onFileUpload= {this.onFileUpload}
                                onImprovementChange= {this.improvementChange}
                                onTilesChange={this.tilesChange}
                                onAlgorithmChange={this.algorithmChange}
                                uploadClickable= {!this.state.hasUploadedImage}
                                display= {this.state.selectedFile !== null}
                            />
                        </Container>
                    </Grid>


                    <Grid item xs= {6}>
                        <header className="App-title">
                            <h1>
                                <span>Mosaic Generator</span>
                            </h1>
                            
                        </header>
                        
                        <header className="App-header">
                            <h3>
                                Upload the image you want to mosaic
                            </h3>
                            <div>
                            <Button variant="contained" component="label" color="primary" startIcon={<CollectionsIcon />}>
                                <input type="file" onChange={this.onFileChange} hidden/>
                                    Choose Image
                            </Button>
                            </div>
                        </header>
                        <Box sx={{marginTop: '2rem'}}>
                            <MosaicDisplay returnedImage= {this.state.returnedImage}/>
                        </Box>
                    </Grid>
                    <Grid item xs= {3}>
                        <AboutMeCard/>
                    </Grid>
                </Grid>
			</div>

		);
	}
}

export default App;

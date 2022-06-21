import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import Button from '@mui/material/Button';
import React, { Component, useEffect } from 'react';
import CollectionsIcon from '@mui/icons-material/Collections';
import ParametersCard from './component/parameters-card/ParametersCard.js'
import MosaicDisplay from './component/mosaic-display/MosaicDisplay.js'
import Grid from '@mui/material/Grid';
import FileData from './component/parameters-card/FileData';
import { Box, Container } from '@mui/material';
import AboutMeCard from './component/about-me/AboutMeCard';
import Resizer from "react-image-file-resizer";

const url = 'https://b-e1.eba-3debdnme.ap-southeast-1.elasticbeanstalk.com/'
//const url = 'http://127.0.0.1:5000/'

function makeid(length) {
    var result           = '';
    var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var charactersLength = characters.length;
    for ( var i = 0; i < length; i++ ) {
      result += characters.charAt(Math.floor(Math.random() * 
 charactersLength));
   }
   return result;
}

const resizeFile = (file) => new Promise(resolve => {
    Resizer.imageFileResizer(file, 1200, 1200, 'JPEG', 80, 0,
    uri => {
      resolve(uri);
    }, 'file' );
});

class App extends Component {
    
    constructor(props) {
        super(props)
        this.routine()
        
    }

	state = {
		// Initially, no file is selected
		selectedFile: null,
		returnedImage: null,
        hasUploadedImage: false,
        algorithm: 'Cosine',
        tiles: 32,
        improvement: 0.25,
        awaitingResponse: false,
        awaitingResponseFilename: null,
        mosaicReady: 0

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
	onFileChange = async (event) => {
        const file = event.target.files[0];
        console.log(file)
        const image = await resizeFile(file);
        const newImage = new File([image], 
            `${makeid(5)}${image.name.slice(-5)}`);
		// Update the state
		this.setState({ selectedFile: newImage});
        console.log(newImage)
	};

	// On file upload (click the upload button)
	onFileUpload = () => {
        // The image is sent to the server
        this.setState({hasUploadedImage: true})
        const prefix = makeid(5)
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
        
        this.setState({awaitingResponseFilename: this.state.selectedFile.name, mosaicReady: 0})

		axios({
			method: 'post',
			url: url.concat('uploader'),
            data: formData,
            timeout:36000000
		}).then((response) => {
			console.log(response);
			this.setState({awaitingResponse: true});
		  }, (error) => {
			console.log(error);
		  });
	};



    get_mosaic = () => {
        const formData = new FormData();
        formData.append('filename', this.state.awaitingResponseFilename)
        axios({
			method: 'post',
			url: url.concat('get_mosaic'),
            data: formData,
			responseType: 'blob',
            timeout:36000000
		}).then((response) => {
			console.log(response);
			this.setState({awaitingResponse: false, hasUploadedImage: false, returnedImage: response.data});
		  }, (error) => {
			console.log(error);
		  });
    }

    mosaic_ready = () => {
        const formData = new FormData();
        formData.append('filename', this.state.awaitingResponseFilename)
        axios({
			method: 'post',
			url: url.concat('mosaic_ready'),
            data: formData,
            timeout:36000000
		}).then((response) => {
            this.setState({mosaicReady: response.data.response})
		  }, (error) => {
			console.log(error);
            return 'no'
		  });

    }

    routine = () => {
        if (this.state.awaitingResponse) {
            this.mosaic_ready()
            console.log(this.state.mosaicReady)
            if (this.state.mosaicReady == 1) {
                this.get_mosaic()
            }
        }

        setTimeout(this.routine, 7000)
    }



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

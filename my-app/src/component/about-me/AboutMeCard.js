import { Card, CardContent, Typography, Stack, Container} from "@mui/material";
import { Component } from "react";
import { } from "@mui/material";

class AboutMeCard extends Component{

    constructor(props) {
        super(props)
    }

    render () {
        return (
            <Container>
                <Card sx={{marginTop: '15px'}} style={{marginTop: '6rem'}}>
                    <Stack spacing={-1}>
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 32, color: '#ffffff'}} align='left'>
                                About me
                            </Typography>
                        </CardContent>
                        
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 16, color: '#ffffff'}} align='left'>
                                Name: Paul
                                <br/>
                                Age: 21
                                <br/>
                                School: NUS
                                <br/>
                                Education: MSc. Computer science
                                
                            </Typography>
                        </CardContent>
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 16, color: '#ffffff'}} align='left'>
                                This is my thesis project to validate my Msc in Computer science.
                                You can find my other projects here: <a href="https://xwkya.github.io" target="_blank" style={{color: 'blue'}}>Portfolio</a>.
                                
                            </Typography>
                        </CardContent>

                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 32, color: '#ffffff'}} align='left'>
                                About this project
                            </Typography>
                        </CardContent>
                        <CardContent className= 'Card'>
                            <Typography sx={{ fontSize: 16, color: '#ffffff'}} align='left'>
                                Project uses multiple methods for tile matching such as pixel average by region or
                                jpeg-like quality matching between tiles and real image.
                                <br/>
                                A neural network has been trained to estimate the matching of images, trading accuracy for speed.
                                <br/><br/>
                                Estimated time: 5s-1m

                            </Typography>
                        </CardContent>
                    </Stack>
                </Card>
            </Container>
        )
    }
}

export default AboutMeCard;
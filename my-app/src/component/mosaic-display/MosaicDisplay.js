import React, { Component } from 'react';
import ReactImageMagnify from 'react-image-magnify';

// Display image given

class MosaicDisplay extends Component {
    constructor(props) {
        super(props)
    }

    render () {
        if (this.props.returnedImage !== null) {
            return (
                
                /*
                <ReactImageMagnify {...{
                    smallImage: {
                        alt: "Test",
                        isFluidWidth: true,
                        src: URL.createObjectURL(this.props.returnedImage),
                    },
                    largeImage: {
                        src: URL.createObjectURL(this.props.returnedImage),
                        width: 2000,
                        height: 2000
                    },
                    enlargedImagePortalId: "myPortal"
                }}/>*/
                
                <img 
                    src={URL.createObjectURL(this.props.returnedImage)}
                    width={'90%'}
                />
                
                
            )
        }
    }
}

export default MosaicDisplay
import { Component } from "react"

class FileData extends Component {
    constructor(props) {
        super(props)
    }

    render () {
        if (this.props.selectedFile !== null) {

            return (
                <img
                    src={URL.createObjectURL(this.props.selectedFile)}
                    style={{width: '100%', marginTop: '10rem'}}
                />
            );
        }
    }
}

export default FileData;
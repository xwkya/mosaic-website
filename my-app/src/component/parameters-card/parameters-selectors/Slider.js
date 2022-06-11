import Slider from '@mui/material/Slider'
import { styled } from '@mui/material/styles';

const CustomSlider = styled(Slider)({
    color: '#6c80ba',
    '& .MuiSlider-track': {
        border: 'none',
      },
    '& .MuiSlider-thumb': {
        height: 24,
        width: 24,
        backgroundColor: '#fff',
        border: '2px solid currentColor',
        '&:focus, &:hover, &.Mui-active, &.Mui-focusVisible': {
            boxShadow: 'inherit',
        },
        '&:before': {
            display: 'none',
        },
    },
    '& .MuiSlider-markLabel': {
        color: '#ffffff'
    },
    '& .MuiSlider-valueLabel': {
        fontSize: 12,
        fontWeight: 'normal',
        top: -6,
        backgroundColor: 'unset',
        color: '#ffffff',
        '&:before': {
          display: 'none',
        },
        '& *': {
          background: 'transparent',
          color: '#fff',
        },
      },
})

export default CustomSlider
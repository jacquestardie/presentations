import React from "react";
import Presentation from "../README.md"

const H1 = props => <h1 style={{ color: 'tomato' }} {...props} />
const Pre = props => <pre style={{ font: 'monospace' }} {...props} />
const P = props => <p style={{ color: 'purple', font-size: "2em" }} {...props} />

  export default () => <Presentation components={{ 
    h1: H1,
    pre: Pre,
    p: P
  }} />

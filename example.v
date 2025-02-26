// Example RTL file - adder_with_mux.v
module adder_with_mux(
    input [7:0] a, b, c, d,
    input sel,
    output [8:0] result
);
    wire [7:0] mux_out;
    wire [8:0] sum;
    
    // Mux operation
    assign mux_out = sel ? c : d;
    
    // Complex operation using mux output
    assign result = (a & b) + (mux_out ^ a);
    
endmodule

# 01/12/2025
* Tests:
  - Test OBB and non-OBB: Good
  - Test OP CUA with string node instead of numeric: Good
    dv = ua.DataValue(ua.Variant(val,ua.VariantType.Float))
    node.set_value(dv)

    self.node_x0 = self.get_node('ns=4;i=1277')\
    self.node_y0 = self.get_node('ns=4;i=1278')\
    self.node_z0 = self.get_node('ns=4;i=1279')\
    
    self.node_x0 = self.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"[0]')\
    self.node_y0 = self.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"[1]')\
    self.node_z0 = self.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"[2]')\


# 29/10/2025
* Bug fix:
  * [Fixed] Multi session with YRC 1000 - fixed by using "with" for initializing OPCUA client
  * Cannot send data from python script to PLC data block tags -> Solution: Use OPCUA interface within TIA Portal to access these data tags
  * Control delay due to constant command pooling
- TODO:
  * Robot cannot move backwards because camera can only see forward --> need a solution
  * Clean the code before move on
  

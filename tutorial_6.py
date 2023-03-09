<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Cliff walking mission based on Sutton and Barto.</Summary>
  </About>

  <ServerSection>
    <ServerInitialConditions>
        <Time><StartTime>1</StartTime></Time>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
      <DrawingDecorator>
        <!-- coordinates for cuboid are inclusive -->
        <DrawCuboid x1="-2" y1="46" z1="-2" x2="7" y2="50" z2="13" type="air" />            <!-- limits of our arena -->
        <DrawCuboid x1="-2" y1="45" z1="-2" x2="7" y2="45" z2="13" type="lava" />           <!-- lava floor -->
        <DrawCuboid x1="1"  y1="45" z1="1"  x2="3" y2="45" z2="16" type="sandstone" />      <!-- floor of the arena -->


        <DrawBlock x="4"  y="45" z="1" type="cobblestone" />    <!-- the starting marker -->
        <DrawBlock x="4"  y="45" z="8" type="lapis_block" />     <!-- the destination marker -->
      </DrawingDecorator>

      <!-- time before program restarts -->
      <ServerQuitFromTimeUp timeLimitMs="20000"/>
      
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Cristina</Name>

    <!-- where to start our agent -->
    <AgentStart>
      <Placement x="4.5" y="46.0" z="1.5" pitch="30" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
      <DiscreteMovementCommands/>
      <ObservationFromFullStats/>

      <!-- code below sets the rewards -->
      <RewardForTouchingBlockType>
        <Block reward="-100.0" type="lava" behaviour="onceOnly"/>
        <Block reward="100.0" type="lapis_block" behaviour="onceOnly"/>
        <Block reward= "1.0"   type="sandstone" behaviour="oncePerBlock"/>
      </RewardForTouchingBlockType>

      <RewardForSendingCommand reward="-2" />

      <!-- when do we quite programs -->
      <AgentQuitFromTouchingBlockType>
          <Block type="lava" />
          <Block type="lapis_block" />
      </AgentQuitFromTouchingBlockType>

    </AgentHandlers>
  </AgentSection>

</Mission>

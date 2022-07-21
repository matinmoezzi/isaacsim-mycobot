# isaacsim-mycobot
MyCobot simulation in IsaacSim

Recommendation: clone this repo into the OmniIsaacGymEnvs/omniisaacgymenvs/

## For mycobot.usd Creating Instancable Assets

```
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable
convert_asset_instanceable(
    asset_usd_path= ‘/~/Desktop/mycobotv1/mycobot.usd’, 
    source_prim_path= ‘/mycobot’, 
    save_as_path= ‘~/Desktop/mycobotv1/mycobot_with_instance_v(x).usd’,
)
```

# Paper scope represented by this repository

This package represents the revised manuscript's decision-system workflow:

1. encode room geometry, window projection, luminaire location, heights, and reflectances;
2. train MResUNet to predict a full electric illuminance field;
3. validate the checkpoint with design-region MAE and SSIM;
4. predict each luminaire contribution and superpose multiple luminaires;
5. combine the current daylight field with candidate electric-light fields;
6. optimize luminaire dimming ratios with particle swarm optimization.

The physical light-field measurement hardware, electrical automation hardware, automatic shading controller, complete raw simulation workflow, and all annual case outputs are outside the compact public package.


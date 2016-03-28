
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic(u'writefile', u'f2.template', u'<!DOCTYPE html>\n\n<meta charset="utf-8">\n<style>\n\n\n.node {\n  stroke: #fff;\n  stroke-width: 0.5px;\n}\n\n.link {\n  stroke: #999;\n  stroke-opacity: .6;\n}\n\n</style>\n<body>\n<script src="//d3js.org/d3.v3.min.js"></script>\n<script>\n\nvar width = 960,\n    height = 500;\n\nvar color = d3.scale.category20();\n\nvar force = d3.layout.force()\n    .charge(-1)\n    .linkDistance(500)\n    .size([width, height]);\n\nvar svg = d3.select("body").append("svg")\n    .attr("width", width)\n    .attr("height", height);\n\nd3.json("connectivity.json", function(error, graph) {\n  if (error) throw error;\n\n  force\n      .nodes(graph.nodes)\n      .links(graph.links)\n      .start();\n\n  var link = svg.selectAll(".link")\n      .data(graph.links)\n    .enter().append("line")\n      .attr("class", "link")\n      .style("stroke-width", function(d) { return Math.sqrt(d.value); });\n\n  var node = svg.selectAll(".node")\n      .data(graph.nodes)\n    .enter().append("circle")\n      .attr("class", "node")\n      .attr("r", 5)\n      .style("fill", function(d) { return color(d.group); })\n      .call(force.drag);\n\n  node.append("title")\n      .text(function(d) { return d.name; });\n\n  force.on("tick", function() {\n    link.attr("x1", function(d) { return d.source.x; })\n        .attr("y1", function(d) { return d.source.y; })\n        .attr("x2", function(d) { return d.target.x; })\n        .attr("y2", function(d) { return d.target.y; });\n\n    node.attr("cx", function(d) { return d.x; })\n        .attr("cy", function(d) { return d.y; });\n  });\n});\n\n</script>')


# In[2]:

from IPython.display import IFrame
import re

def replace_all(txt,d):
    rep = dict((re.escape('{'+k+'}'), str(v)) for k, v in d.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)    

count=0
def serve_html(s,w,h):
    import os
    global count
    count+=1
    fn= '__tmp'+str(os.getpid())+'_'+str(count)+'.html'
    with open(fn,'w') as f:
        f.write(s)
    return IFrame('files/'+fn,w,h)

def f2(w=500,h=400):
    '''
    d={
       'width'      :w,
       'height'     :h,
       'ball_count' :ball_count,
       'rad_min'    :rad_min,
       'rad_fac'    :rad_fac,
       'color_count':color_count
       }
       '''
    with open('f2.template','r') as f:
        s=f.read()
    #s= replace_all(s,d)        
    return serve_html(s,w+30,h+30)


# In[3]:

f2(w=2000,h=1600)


# In[ ]:




# In[ ]:




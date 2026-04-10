const API = window.location.origin + '/api/v1';
function fmtN(v,d=4){if(v==null||typeof v!=='number'||isNaN(v)||!isFinite(v))return'-';const a=Math.abs(v);if(a===0)return'0';if(a>=1e6||a<1e-3)return v.toExponential(2);if(a>=100)return v.toFixed(1);if(a>=1)return v.toFixed(d>3?3:d);return v.toFixed(d)}
const PL = {paper_bgcolor:'transparent',plot_bgcolor:'rgba(26,29,39,0.5)',font:{family:'JetBrains Mono',size:10,color:'#94a3b8'},margin:{l:50,r:16,t:24,b:56},xaxis:{gridcolor:'#2a2d3a',zerolinecolor:'#2a2d3a'},yaxis:{gridcolor:'#2a2d3a',zerolinecolor:'#2a2d3a'},legend:{orientation:'h',y:-0.3,yanchor:'top',font:{size:9}},modebar:{bgcolor:'transparent'}};
const PC = {responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
const CL = ['#00d4aa','#3b82f6','#f59e0b','#ef4444','#a78bfa','#ec4899','#06b6d4','#84cc16'];
const GC = {CenterOfSearchSpace:'#64748b',Sobol:'#3b82f6',MBM:'#00d4aa',bo:'#00d4aa',sobol:'#3b82f6'};

function app(){return{
  tab:'overview',tabs:[{id:'overview',label:'Overview'},{id:'trials',label:'Trials'},{id:'objectives',label:'Objectives'},{id:'analysis',label:'Analysis'},{id:'streaming',label:'Streaming'},{id:'dependencies',label:'Dependencies'},{id:'predict',label:'Predict'},{id:'config',label:'Config'}],
  state:{status:null,experiment:null,trials:null,objectives:null,streaming:null,generation:null,pareto:null,events:null},
  lastUpdate:'',tf:'all',tsk:'index',tsd:1,xt:null,td:null,tdViz:null,tdVizLoading:false,tdVizErr:'',pvState:'',pp:{},predicting:false,pr:null,pe:null,cs:[],cv:{},cmp:[],
  enlargedEl:null,
  anlMetric:'',anlTopK:10,anlLoading:false,anlError:'',anlSens:null,anlCV:null,anlPC:null,anlContour:null,anlSearch:null,anlInsights:null,anlHealth:null,anlGroupSens:null,anlGroupInt:null,anlRobust:null,anlGroupBest:null,frozenGroups:{},fetchTrialIdx:'',
  sweeping:false,sweepData:null,sweepErr:null,sweepNPts:25,
  _schedPoll(){setTimeout(async()=>{await this.r();this._schedPoll()},3000)},
  async init(){await this.r();this._schedPoll();this.$watch('tab',()=>this.r());document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const n=parseInt(e.key);if(n>=1&&n<=this.tabs.length)this.tab=this.tabs[n-1].id;if(e.key==='v'&&!e.ctrlKey&&!e.metaKey){const hovered=document.querySelector('.chart-card:hover');if(hovered){const plotDiv=hovered.querySelector('.js-plotly-plot')||(hovered.classList.contains('js-plotly-plot')?hovered:null);if(plotDiv&&plotDiv.data){this.enlargedEl=true;this.$nextTick(()=>{const target=document.getElementById('enlarge-target');if(target)Plotly.react(target,plotDiv.data,{...plotDiv.layout,width:null,height:null},PC)})}}}if(e.key==='Escape'){this.enlargedEl=null;const et=document.getElementById('enlarge-target');if(et)Plotly.purge(et)}});this.$watch('xt',async(v)=>{this.td=null;this.tdViz=null;this.tdVizLoading=false;this.tdVizErr='';if(v!==null){const r=await this.f('trials/'+v);if(r&&!r._error)this.td=r}})},
  _etags:{},
  _tabEps:{overview:['generation','events'],trials:[],objectives:['pareto'],analysis:[],streaming:['streaming'],dependencies:[],predict:[],config:['config']},
  async r(){
    // Always fetch core data; add tab-specific endpoints
    const base=['status','experiment','trials','objectives'];
    const te=this._tabEps[this.tab]||[];
    const ep=[...new Set([...base,...te])];
    const rs=await Promise.allSettled(ep.map(e=>this.f(e)));
    ep.forEach((e,i)=>{const v=rs[i].status==='fulfilled'?rs[i].value:null;if(v&&!v._error&&!v._304)this.state[e]=v});
    this.lastUpdate=new Date().toLocaleTimeString();
    if(this.state.experiment?.parameters&&!Object.keys(this.pp).length)for(const p of this.state.experiment.parameters)if(!p.fixed&&p.bounds)this.pp[p.name]=p.parameter_type==='int'?Math.round((p.bounds[0]+p.bounds[1])/2):(p.bounds[0]+p.bounds[1])/2;
    if(!this.anlMetric&&this.state.experiment?.objectives?.length)this.anlMetric=this.state.experiment.objectives[0].name;
    if(!this.cs.length){const s=await this.f('config/schema');if(s?.fields&&!s._error)this.cs=s.fields;const c=await this.f('config');if(c&&!c._error&&!c._304)this.cv=c}
    this.$nextTick(()=>this.charts())
  },
  async f(p,m='GET',b=null){try{const o={method:m,headers:{Accept:'application/json'}};if(m==='GET'&&this._etags[p])o.headers['If-None-Match']=this._etags[p];if(b){o.headers['Content-Type']='application/json';o.body=JSON.stringify(b)}const r=await fetch(API+'/'+p,o);if(r.status===304)return{_304:true};const etag=r.headers.get('etag');if(etag)this._etags[p]=etag;const j=await r.json().catch(()=>null);if(!r.ok)return{_error:j?.detail||j?.error||r.statusText,_status:r.status};return j}catch(e){return{_error:e.message}}},
  allMetricNames(){
    const objNames=(this.state.experiment?.objectives||[]).map(o=>o.name);
    const all=new Set(objNames);
    for(const t of this.state.trials?.trials||[])for(const m of Object.keys(t.metrics||{}))all.add(m);
    // Objectives first, then non-objectives sorted
    const nonObj=[...all].filter(n=>!objNames.includes(n)).sort();
    return[...objNames,...nonObj];
  },
  isObjective(m){return(this.state.experiment?.objectives||[]).some(o=>o.name===m)},
  fTrials(){let t=[...(this.state.trials?.trials||[])];if(this.tf!=='all')t=t.filter(x=>x.status===this.tf);return t.sort((a,b)=>(a[this.tsk]>b[this.tsk]?1:-1)*this.tsd)},
  tsort(k){if(this.tsk===k)this.tsd*=-1;else{this.tsk=k;this.tsd=1}},
  gtp(i){return this.state.trials?.trials?.find(t=>t.index===i)?.parameters||{}},
  async fetchViz(){if(this.xt==null||this.tdVizLoading)return;this.tdVizLoading=true;this.tdVizErr='';const r=await this.f('trials/'+this.xt+'/visualization');if(r?.image)this.tdViz=r.image;else{this.tdViz=null;this.tdVizErr=r?._error||r?.error||'Visualization unavailable'}this.tdVizLoading=false},
  async uploadPvState(ev){const file=ev.target.files?.[0];if(!file)return;try{const body=await file.arrayBuffer();const r=await fetch(API+'/paraview-state',{method:'POST',body});const j=await r.json();if(j?.status==='ok')this.pvState=file.name;else this.tdVizErr=j?.error||'Upload failed'}catch(e){this.tdVizErr=e.message}},
  async removePvState(){await fetch(API+'/paraview-state',{method:'DELETE'});this.pvState=''},
  isBest(t,oName){const obj=this.state.objectives?.objectives?.[oName];if(!obj?.best)return false;return t.index===obj.best.trial},
  fmtDur(s){if(s==null)return'-';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;return h?(h+'h '+(m+'m').padStart(3,'0')):m?(m+'m '+(sec+'s').padStart(3,'0')):(sec+'s')},
  avgDuration(){const ts=this.state.trials?.trials||[];const vals=ts.filter(t=>t.execution_time_s!=null&&t.status!=='RUNNING').map(t=>t.execution_time_s);return vals.length?this.fmtDur(Math.round(vals.reduce((a,b)=>a+b,0)/vals.length)):'-'},
  toggleCmp(idx){const i=this.cmp.indexOf(idx);if(i>=0)this.cmp.splice(i,1);else if(this.cmp.length<3)this.cmp.push(idx)},
  cmpDiffers(key){const vals=this.cmp.map(i=>{const t=this.state.trials?.trials?.find(x=>x.index===i);return t?.parameters?.[key]});return vals.some(v=>v!==vals[0])},
  bv(n){const v=this.state.objectives?.objectives?.[n]?.best?.value;return v!=null?fmtN(v):'-'},
  bt(n){return this.state.objectives?.objectives?.[n]?.best?.trial??'-'},
  opairs(){const n=(this.state.experiment?.objectives||[]).map(o=>o.name);const p=[];for(let i=0;i<n.length;i++)for(let j=i+1;j<n.length;j++)p.push([n[i],n[j]]);return p},
  genW(n){const c=this.state.generation?.node_counts||{};const tg=this.state.generation?.node_targets||{};const cur=c[n]||0;const max=tg[n];if(max!=null&&max>0)return Math.min(100,cur/max*100)+'%';const t=Object.values(c).reduce((a,b)=>a+b,0);return t?(cur/t*100)+'%':'0%'},
  genC(n){return GC[n]||CL[Object.keys(this.state.generation?.node_counts||{}).indexOf(n)%CL.length]},
  genLabel(n){const c=this.state.generation?.node_counts||{};const tg=this.state.generation?.node_targets||{};const cur=c[n]||0;const max=tg[n];return max!=null?cur+' / '+max:cur+' trials'},
  pct(k){const t=this.state.status?.trials_total||0;const v=this.state.status?.[k]||0;return t?Math.round(v/t*100)+'%':''},
  pctc(s){const t=this.state.status?.trials_total||0;const v=this.state.trials?.counts?.[s]||0;return t?Math.round(v/t*100)+'%':''},
  budgetPct(){const t=this.state.status?.trials_total||0;const m=this.state.experiment?.max_trials||0;return m?Math.round(t/m*100)+'% used':''},
  getDepNames(){
    const names=new Set();
    for(const t of this.state.trials?.trials||[])
      for(const d of (t.dependencies||[]))if(d.name)names.add(d.name);
    return[...names].sort();
  },
  renderDepGraph(){
    const ts=this.state.trials?.trials||[];
    if(!ts.length)return;
    const statusColor={'COMPLETED':'#22c55e','RUNNING':'#3b82f6','FAILED':'#ef4444','EARLY_STOPPED':'#f59e0b','ABANDONED':'#64748b'};
    const sm={};ts.forEach(t=>{sm[t.index]=t});
    const mkHover=(n)=>{
      const trial=sm[n];if(!trial)return'T'+n;
      let txt='<b>T'+n+'</b> ['+trial.status+']';
      if(trial.gen_node)txt+=' <i>'+trial.gen_node+'</i>';
      txt+='<br>';
      const params=trial.parameters||{};const pk=Object.keys(params).slice(0,5);
      if(pk.length){txt+=pk.map(k=>k+'='+fmtN(params[k])).join(', ');if(Object.keys(params).length>5)txt+=', ...';txt+='<br>'}
      const met=trial.metrics||{};const mk=Object.keys(met);
      if(mk.length)txt+=mk.map(k=>'<b>'+k+'</b>='+fmtN(met[k])).join(', ');
      return txt;
    };
    // Render one graph per dependency name
    for(const depName of this.getDepNames()){
      const el=document.getElementById('dep-graph-'+depName);if(!el)continue;
      // Collect edges for this dependency
      const edges=[];
      for(const t of ts)for(const d of (t.dependencies||[]))
        if(d.name===depName)edges.push([d.source_index,t.index]);
      if(!edges.length)continue;
      const nodeSet=new Set();edges.forEach(([s,t])=>{nodeSet.add(s);nodeSet.add(t)});
      const nodes=[...nodeSet].sort((a,b)=>a-b);
      // Tree layout
      const children={};const parentOf={};
      edges.forEach(([s,t])=>{if(!children[s])children[s]=[];children[s].push(t);parentOf[t]=s});
      const roots=nodes.filter(n=>!(n in parentOf));
      const pos={};const visited=new Set();let row=0;
      const place=(n,depth)=>{
        if(visited.has(n))return;visited.add(n);
        pos[n]={x:depth*3,y:row};
        const kids=(children[n]||[]).sort((a,b)=>a-b);
        if(!kids.length){row++;return}
        kids.forEach(c=>place(c,depth+1));
      };
      roots.sort((a,b)=>a-b).forEach(r=>place(r,0));
      // Place any unvisited nodes (cycles)
      nodes.filter(n=>!visited.has(n)).forEach(n=>{pos[n]={x:0,y:row++}});
      // Traces
      const annotations=edges.map(([s,t])=>{
        if(!pos[s]||!pos[t])return null;
        const dx=pos[t].x-pos[s].x,dy=pos[t].y-pos[s].y;
        const len=Math.sqrt(dx*dx+dy*dy);if(len===0)return null;
        const r=0.5;// offset in data units (~circle radius)
        const ux=dx/len,uy=dy/len;
        return{ax:pos[s].x+ux*r,ay:pos[s].y+uy*r,x:pos[t].x-ux*r,y:pos[t].y-uy*r,
          xref:'x',yref:'y',axref:'x',ayref:'y',
          showarrow:true,arrowhead:3,arrowsize:1.2,arrowwidth:1.5,arrowcolor:'#4a5568'};
      }).filter(Boolean);
      const traces=[{
        x:nodes.map(n=>pos[n].x),y:nodes.map(n=>pos[n].y),mode:'markers+text',
        marker:{size:28,color:nodes.map(n=>statusColor[sm[n]?.status]||'#64748b'),line:{color:'#1a1a2e',width:2}},
        text:nodes.map(n=>'T'+n),textposition:'middle center',textfont:{size:9,color:'#fff'},
        hovertext:nodes.map(n=>mkHover(n)),hoverinfo:'text',showlegend:false
      }];
      Plotly.react(el,traces,{...PL,
        xaxis:{...PL.xaxis,showgrid:false,zeroline:false,showticklabels:false,fixedrange:false},
        yaxis:{...PL.yaxis,showgrid:false,zeroline:false,showticklabels:false,autorange:'reversed',fixedrange:false},
        annotations,margin:{l:10,r:10,t:10,b:10},hovermode:'closest',dragmode:'pan'},PC);
    }
  },
  async pickPoint(objective){
    const r=await this.f('pareto/pick?objective='+encodeURIComponent(objective));
    if(r?.parameters){
      this.pp={...this.pp,...r.parameters};
      // Update input fields
      for(const[k,v]of Object.entries(r.parameters)){const el=document.getElementById('pinp-'+k);if(el)el.value=v}
      // Show predictions from the pick response
      if(r.predictions)this.pr={predictions:r.predictions};
    }
  },
  resetParams(){
    for(const p of this.state.experiment?.parameters||[]){
      if(!p.fixed&&p.bounds&&!this.isParamFrozen(p.name)){
        const v=p.parameter_type==='int'?Math.round((p.bounds[0]+p.bounds[1])/2):(p.bounds[0]+p.bounds[1])/2;
        this.pp[p.name]=v;
        const el=document.getElementById('pinp-'+p.name);if(el)el.value=v;
      }
    }
    this.pr=null;this.pe=null;
  },
  getGroups(){
    const pg=this.state.experiment?.parameter_groups||{};
    const gs=new Set();
    for(const groups of Object.values(pg))for(const g of groups)gs.add(g);
    return[...gs].sort();
  },
  toggleFreezeGroup(group){this.frozenGroups={...this.frozenGroups,[group]:!this.frozenGroups[group]}},
  isParamFrozen(pname){
    const groups=this.state.experiment?.parameter_groups?.[pname]||[];
    return groups.some(g=>this.frozenGroups[g]);
  },
  async fetchTrialParams(){
    const idx=parseInt(this.fetchTrialIdx);
    if(isNaN(idx))return;
    const t=this.state.trials?.trials?.find(t=>t.index===idx);
    if(t?.parameters){for(const[k,v]of Object.entries(t.parameters)){this.pp[k]=v;const el=document.getElementById('pinp-'+k);if(el)el.value=v}}
  },
  async fetchGroupBest(group){
    const pg=this.state.experiment?.parameter_groups||{};
    const groupParams=Object.entries(pg).filter(([_,gs])=>gs.includes(group)).map(([n])=>n);
    const values={};
    for(const p of groupParams)if(this.pp[p]!==undefined)values[p]=this.pp[p];
    const r=await this.f('analysis/group-conditional-best','POST',{group,values});
    if(r?.non_group_params){
      for(const[k,v]of Object.entries(r.non_group_params)){this.pp[k]=v;const el=document.getElementById('pinp-'+k);if(el)el.value=v}
      this.anlGroupBest=r;
    }
  },
  uncertaintyBar(pred){
    if(!pred?.mean||!pred?.sem)return'0%';
    const ratio=Math.min(Math.abs(pred.sem/pred.mean)*10,1);
    return(100-ratio*100)+'%';
  },
  async predict(){this.predicting=true;this.pe=null;this.pr=null;try{const r=await this.f('predict','POST',{parameters:this.pp});if(r?._error)this.pe=r._error;else if(r?.predictions)this.pr=r;else this.pe='No predictions returned'}catch(e){this.pe=e.message}this.predicting=false},
  async sweepGroup(group){
    this.sweeping=true;this.sweepErr=null;this.sweepData=null;
    try{
      const r=await this.f('predict/group-sweep','POST',{frozen_group:group,base_parameters:this.pp,n_points:this.sweepNPts});
      if(r?._error)this.sweepErr=r._error;else{this.sweepData=r;if(r.base_predictions)this.pr={predictions:r.base_predictions}}
    }catch(e){this.sweepErr=e.message}
    this.sweeping=false;
    this.$nextTick(()=>this.renderSweepCharts());
  },
  renderSweepCharts(){
    if(!this.sweepData?.curves)return;
    const objs=this.state.experiment?.objectives||[];
    this.sweepData.curves.forEach((curve,idx)=>{
      const el=document.getElementById('sweep-chart-'+idx);
      if(!el)return;
      const traces=[];
      objs.forEach((obj,oi)=>{
        const mn=obj.name;const pred=curve.predictions[mn];if(!pred)return;
        const color=CL[oi%CL.length];
        const upper=pred.mean.map((m,i)=>m+pred.sem[i]);
        const lower=pred.mean.map((m,i)=>m-pred.sem[i]);
        // Uncertainty band (SEM area) — single closed polygon
        const hr=parseInt(color.slice(1,3),16),hg=parseInt(color.slice(3,5),16),hb=parseInt(color.slice(5,7),16);
        const fc='rgba('+hr+','+hg+','+hb+',0.18)';
        const bandX=[...curve.x_values,...[...curve.x_values].reverse()];
        const bandY=[...upper,...[...lower].reverse()];
        traces.push({x:bandX,y:bandY,fill:'toself',fillcolor:fc,line:{width:0},mode:'none',showlegend:false,legendgroup:mn,hoverinfo:'skip'});
        // Mean line with SEM in hover
        const htext=pred.mean.map((m,i)=>mn+': '+fmtN(m)+' ± '+fmtN(pred.sem[i]));
        traces.push({x:curve.x_values,y:pred.mean,mode:'lines',line:{color,width:2},name:mn,
          showlegend:idx===0,legendgroup:mn,text:htext,hoverinfo:'text+x'});
        // Current value marker
        const bv=this.pp[curve.param_name];const bp=this.sweepData.base_predictions?.[mn];
        if(bv!==undefined&&bp){traces.push({x:[bv],y:[bp.mean],mode:'markers',marker:{color,size:8,symbol:'diamond'},showlegend:false,legendgroup:mn,name:'Current'})}
      });
      const layout={...PL,title:{text:curve.param_name,font:{size:12}},
        xaxis:{...PL.xaxis,title:curve.param_name},yaxis:{...PL.yaxis,title:'Objective'},
        legend:{orientation:'h',y:-0.25,yanchor:'top',font:{size:9}},hovermode:'x unified',margin:{l:55,r:15,t:30,b:55}};
      Plotly.react(el,traces,layout,PC);
      // Sync legend clicks across all sweep charts
      if(idx===0){
        const syncLegend=(evtData)=>{
          const ci=evtData.curveNumber;
          const firstEl=document.getElementById('sweep-chart-0');
          if(!firstEl||!firstEl.data||!firstEl.data[ci])return;
          const vis=firstEl.data[ci].visible;
          const grp=firstEl.data[ci].legendgroup;
          if(!grp)return;
          this.sweepData.curves.forEach((_,j)=>{
            if(j===0)return;
            const oel=document.getElementById('sweep-chart-'+j);
            if(!oel||!oel.data)return;
            const upd={};
            oel.data.forEach((t,ti)=>{if(t.legendgroup===grp)upd[ti]=vis});
            if(Object.keys(upd).length){
              const indices=Object.keys(upd).map(Number);
              Plotly.restyle(oel,{visible:indices.map(i=>upd[i])},indices);
            }
          });
        };
        el.on('plotly_legendclick',(d)=>{setTimeout(()=>syncLegend(d),50);return true});
        el.on('plotly_legenddoubleclick',(d)=>{setTimeout(()=>syncLegend(d),50);return true});
      }
    });
  },
  async generateInsights(){
    this.anlLoading=true;this.anlError='';
    const m=this.anlMetric||(this.state.experiment?.objectives?.[0]?.name)||'';
    if(!m){this.anlError='No metric selected';this.anlLoading=false;return}
    const isRobust=!!this.state.experiment?.robust_optimization;
    const [s,pc,cv,ct,ss,ins,hc,gs,gi,rp_]=await Promise.allSettled([
      this.f('analysis/sensitivity','POST',{metric:m,top_k:this.anlTopK}),
      this.f('analysis/parallel-coordinates','POST',{metric:m}),
      this.f('analysis/cross-validation','POST',{}),
      this.f('analysis/contour','POST',{metric:m}),
      this.f('analysis/search-space','POST',{}),
      this.f('analysis/insights','POST',{}),
      this.f('analysis/healthchecks','POST',{}),
      this.f('analysis/group-sensitivity','POST',{metric:m}),
      this.f('analysis/group-interactions','POST',{metric:m}),
      isRobust?this.f('analysis/robustness-profile','POST',{}):Promise.resolve(null),
    ]);
    const _r=(r)=>r.status==='fulfilled'&&r.value&&!r.value._error?r.value:null;
    const _e=(r)=>r.status==='fulfilled'&&r.value?._error?r.value._error:'';
    this.anlSens=_r(s)?.figure?_r(s):null;
    this.anlPC=_r(pc)?.figure?_r(pc):null;
    this.anlCV=_r(cv)?.figure?_r(cv):null;
    this.anlContour=_r(ct);
    this.anlSearch=_r(ss);
    this.anlInsights=_r(ins);
    this.anlHealth=_r(hc);
    const gsR=_r(gs);this.anlGroupSens=gsR?.groups?gsR:null;
    const giR=_r(gi);this.anlGroupInt=giR?.matrix?giR:null;
    const rpR=_r(rp_);this.anlRobust=rpR?.per_context?rpR:null;
    // Failed analyses are silently skipped — panels just don't appear
    this.anlLoading=false;
    this.$nextTick(()=>{
      const DL={...PL,paper_bgcolor:'transparent',plot_bgcolor:'rgba(26,29,39,0.5)'};
      const rp=(id,fig)=>{const el=document.getElementById(id);if(el&&fig)Plotly.react(el,fig.data,{...DL,...fig.layout},PC)};
      rp('anl-sens',this.anlSens?.figure);
      rp('anl-cv',this.anlCV?.figure);
      // Parallel coordinates: reorder dimensions by group if available
      if(this.anlPC?.figure){
        const fig=this.anlPC.figure;
        const pg=this.state.experiment?.parameter_groups||{};
        if(fig.data?.[0]?.dimensions&&Object.keys(pg).length){
          const dims=fig.data[0].dimensions;
          dims.sort((a,b)=>{const ga=(pg[a.label]||['zzz'])[0];const gb=(pg[b.label]||['zzz'])[0];return ga<gb?-1:ga>gb?1:0});
        }
        rp('anl-pc',fig);
      }
      if(this.anlContour?.cards){this.anlContour.cards.forEach((c,i)=>rp('anl-cont-'+i,c.figure))}
      else if(this.anlContour?.figure){rp('anl-cont-0',this.anlContour.figure)}
      // Group sensitivity bar chart
      if(this.anlGroupSens?.groups){
        const el=document.getElementById('anl-gsens');
        if(el){
          const g=this.anlGroupSens.groups;
          const names=Object.keys(g).sort((a,b)=>g[b]-g[a]);
          const vals=names.map(n=>g[n]);
          Plotly.react(el,[{y:names,x:vals,type:'bar',orientation:'h',marker:{color:names.map((_,i)=>CL[i%CL.length])}}],{...DL,margin:{...DL.margin,l:120}},PC);
        }
      }
      // Group interaction heatmap
      if(this.anlGroupInt?.matrix){
        const el=document.getElementById('anl-gint');
        if(el){
          const d=this.anlGroupInt;
          Plotly.react(el,[{z:d.matrix,x:d.groups,y:d.groups,type:'heatmap',colorscale:[[0,'#1a1a2e'],[0.5,'#3b82f6'],[1,'#ef4444']],showscale:true}],{...DL,margin:{...DL.margin,l:120,b:80}},PC);
        }
      }
      // Robustness profile: grouped bar chart (per metric, per context point)
      if(this.anlRobust?.per_context){
        const el=document.getElementById('anl-robust');
        if(el){
          const d=this.anlRobust;
          const metrics=Object.keys(d.per_context[0]?.predictions||{});
          const ctxLabels=d.per_context.map((_,i)=>'ctx#'+i);
          const traces=[];
          metrics.forEach((mn,mi)=>{
            const vals=d.per_context.map(pc=>pc.predictions[mn]?.mean||0);
            const errs=d.per_context.map(pc=>pc.predictions[mn]?.sem||0);
            traces.push({
              name:mn,x:ctxLabels,y:vals,error_y:{type:'data',array:errs,visible:true},
              type:'bar',marker:{color:CL[mi%CL.length]}
            });
          });
          // Add CVaR reference line per metric
          metrics.forEach((mn,mi)=>{
            if(d.cvar&&d.cvar[mn]!=null){
              traces.push({
                name:'CVaR('+mn+')',x:[ctxLabels[0],ctxLabels[ctxLabels.length-1]],
                y:[d.cvar[mn],d.cvar[mn]],mode:'lines',
                line:{color:CL[mi%CL.length],dash:'dash',width:2},
                showlegend:true
              });
            }
          });
          Plotly.react(el,traces,{...DL,barmode:'group',
            xaxis:{title:'Context Point'},yaxis:{title:'Predicted Value'},
            margin:{...DL.margin,b:60}},PC);
        }
      }
      // Render figure cards from search-space, insights, healthchecks
      for(const[prefix,src]of[['anl-ss-',this.anlSearch],['anl-ins-',this.anlInsights],['anl-hc-',this.anlHealth]]){
        if(src?.cards)src.cards.forEach((c,i)=>{if(c.figure)rp(prefix+i,c.figure)});
      }
    });
  },
  gcfg(){const g={};for(const f of this.cs){const s=f.path.split('.')[0];if(!g[s])g[s]=[];g[s].push(f)}
    const order=['experiment','optimization','trial_generation','orchestration_settings','store','baseline','trial_dependencies','existing_trials'];
    const sorted={};for(const k of order){if(g[k])sorted[k]=g[k]}for(const k in g){if(!sorted[k])sorted[k]=g[k]}return sorted},
  isComplex(v){return v!==null&&typeof v==='object'},
  _rv(v,mutable,indent){
    const c=mutable?'color:var(--accent)':'color:var(--pico-secondary)';
    const ml=indent+'rem';
    if(v===null||v===undefined)return'<span style="'+c+'">null</span>';
    if(typeof v!=='object')return'<span style="'+c+'">'+v+'</span>';
    if(Array.isArray(v)){
      if(!v.length)return'<span style="'+c+'">[]</span>';
      let h='';
      v.forEach((item)=>{
        if(typeof item==='object'&&item!==null){
          h+='<div style="margin-left:'+ml+';'+c+'">-</div>';
          Object.entries(item).forEach(([k,iv])=>{
            const nested=typeof iv==='object'&&iv!==null;
            if(nested){
              h+='<div style="margin-left:'+(indent+0.8)+'rem;'+c+'">'+k+':</div>';
              h+=this._rv(iv,mutable,indent+1.6);
            }else{
              h+='<div style="margin-left:'+(indent+0.8)+'rem;padding-left:'+(k.length+2)+'ch;text-indent:-'+(k.length+2)+'ch;'+c+'">'+k+': '+iv+'</div>';
            }
          });
        }else{h+='<div style="margin-left:'+ml+';'+c+'">- '+item+'</div>'}
      });
      return h;
    }
    let h='';
    Object.entries(v).forEach(([k,dv])=>{
      const nested=typeof dv==='object'&&dv!==null;
      if(nested){
        h+='<div style="margin-left:'+ml+';'+c+'">'+k+':</div>';
        h+=this._rv(dv,mutable,indent+0.8);
      }else{
        h+='<div style="margin-left:'+ml+';padding-left:'+(k.length+2)+'ch;text-indent:-'+(k.length+2)+'ch;'+c+'">'+k+': '+dv+'</div>';
      }
    });
    return h;
  },
  renderYaml(v,mutable,indent=1){
    if(typeof v!=='object'||v===null)return this._rv(v,mutable,0);
    return':<br>'+this._rv(v,mutable,indent);
  },
  cfgMsg:'',cfgOk:false,
  async applyCfg(){const p={};for(const f of this.cs)if(f.mutable&&this.cv[f.path]!==undefined)p[f.path]=this.cv[f.path];if(!Object.keys(p).length){this.cfgMsg='No mutable fields to update';this.cfgOk=false;return}const r=await this.f('config','PATCH',p);if(!r){this.cfgMsg='Failed to reach server';this.cfgOk=false}else if(r.rejected?.length&&!r.updated?.length){this.cfgMsg='Rejected: '+r.rejected.map(x=>x.path+' ('+x.reason+')').join(', ');this.cfgOk=false}else{this.cfgMsg='Updated: '+r.updated.join(', ')+(r.rejected?.length?' | Rejected: '+r.rejected.map(x=>x.path).join(', '):'');this.cfgOk=!r.rejected?.length}setTimeout(()=>this.cfgMsg='',5000)},
  _preserveAndReact(el,traces,layout,config){
    // Capture legend visibility before react, restore after
    const vis={};
    if(el.data)el.data.forEach((t,i)=>{if(t.visible!==undefined&&t.visible!==true)vis[t.name||i]=t.visible});
    Plotly.react(el,traces,layout,config);
    if(Object.keys(vis).length)el.data.forEach((t,i)=>{const v=vis[t.name||i];if(v!==undefined)t.visible=v});
    if(Object.keys(vis).length)Plotly.redraw(el);
  },
  charts(){
    const tmap={};(this.state.trials?.trials||[]).forEach(t=>tmap[t.index]=t.gen_node||'');
    for(const o of this.state.experiment?.objectives||[]){const el=document.getElementById('co-'+o.name);if(!el)continue;const d=this.state.objectives?.objectives?.[o.name];if(!d?.values?.length)continue;const x=d.values.map(v=>v.trial),y=d.values.map(v=>v.value),bs=d.best_so_far||[];
      const colors=x.map(t=>GC[tmap[t]]||'#3b82f6');
      const tr=[{x,y,mode:'markers',name:'observed',marker:{color:colors,size:6},text:x.map(t=>tmap[t]||'')}];
      if(bs.length===y.length)tr.push({x,y:bs,mode:'lines',name:'best-so-far',line:{color:'#00d4aa',width:2}});
      this._preserveAndReact(el,tr,{...PL,xaxis:{...PL.xaxis,title:'Trial',dtick:1}},PC)}
    const hv=document.getElementById('c-hv');if(hv&&this.state.pareto?.hypervolume_trace?.length>1){const h=this.state.pareto.hypervolume_trace;this._preserveAndReact(hv,[{x:h.map(v=>v.trial),y:h.map(v=>v.value),mode:'lines+markers',line:{color:'#00d4aa'},marker:{size:4}}],{...PL,xaxis:{...PL.xaxis,title:'Trial'}},PC)}
    for(const p of this.opairs()){const el=document.getElementById('cp-'+p[0]+'-'+p[1]);if(!el)continue;const d0=this.state.objectives?.objectives?.[p[0]]?.values||[],d1=this.state.objectives?.objectives?.[p[1]]?.values||[];if(d0.length<2)continue;const m0={},m1={};d0.forEach(v=>m0[v.trial]=v.value);d1.forEach(v=>m1[v.trial]=v.value);const cm=Object.keys(m0).filter(t=>t in m1).map(Number);const tr=[{x:cm.map(t=>m0[t]),y:cm.map(t=>m1[t]),mode:'markers',name:'trials',marker:{color:'#3b82f6',size:5},text:cm.map(t=>'T'+t)}];const fr=this.state.pareto?.frontier||[];if(fr.length)tr.push({x:fr.map(f=>f.metrics[p[0]]).filter(v=>v!=null),y:fr.map(f=>f.metrics[p[1]]).filter(v=>v!=null),mode:'markers',name:'pareto',marker:{color:'#00d4aa',size:9,symbol:'diamond'}});this._preserveAndReact(el,tr,{...PL,xaxis:{...PL.xaxis,title:p[0]},yaxis:{...PL.yaxis,title:p[1]}},PC)}
    for(const[mn,pt]of Object.entries(this.state.streaming?.metrics||{})){const el=document.getElementById('cs-'+mn);if(!el)continue;const tr=[];let i=0;for(const[ti,td]of Object.entries(pt)){if(td.steps?.length<2)continue;tr.push({x:td.steps,y:td.values,mode:'lines',name:'T'+ti,line:{color:CL[i%CL.length],width:1.5}});i++}const sh=[];const esTrials=this.state.trials?.trials?.filter(t=>t.status==='EARLY_STOPPED')||[];for(const est of esTrials){const td=pt[est.index];if(td?.steps?.length){const lastStep=td.steps[td.steps.length-1];sh.push({type:'line',x0:lastStep,x1:lastStep,y0:0,y1:1,yref:'paper',line:{color:'#f59e0b',width:2,dash:'dashdot'}})}}const th=this.state.streaming?.thresholds?.[mn];if(th){if(th.type==='threshold'&&th.value!=null){sh.push({type:'line',y0:th.value,y1:th.value,x0:0,x1:1,xref:'paper',line:{color:'#ef4444',dash:'dash',width:2}});tr.push({x:[],y:[],mode:'lines',name:'threshold: '+th.value,line:{color:'#ef4444',dash:'dash'},showlegend:true,hoverinfo:'skip'})}if(th.type==='percentile'&&th.resolved_steps?.length){tr.push({x:th.resolved_steps,y:th.resolved_values,mode:'lines',name:'P'+th.percentile+' threshold',line:{color:'#ef4444',dash:'dash',width:2}})}if(th.min_progression)sh.push({type:'line',x0:th.min_progression,x1:th.min_progression,y0:0,y1:1,yref:'paper',line:{color:'#f59e0b',dash:'dot',width:1.5}})}this._preserveAndReact(el,tr,{...PL,shapes:sh,xaxis:{...PL.xaxis,title:'Step'},yaxis:{...PL.yaxis,title:mn}},PC)}
    this.renderDepGraph();
  }
}}
